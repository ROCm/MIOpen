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
s_getpc_b64 [s6, s7]
label_1:
label_2:
s_load_dwordx2 [s22, s23], [s2, s3], 0
s_load_dwordx16 [s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, s34, s35, s36, s37, s38, s39], [s2, s3], 8
s_load_dwordx8 [s40, s41, s42, s43, s44, s45, s46, s47], [s2, s3], 72
v_readfirstlane_b32 s1, v0
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
s_lshr_b32 s1, s1, 6
s_cmp_ge_u32 s1, 8
s_cbranch_scc0 label_3
s_setprio 3
label_3:
s_waitcnt lgkmcnt(0)
s_and_b32 s28, s28, 262143
s_and_b32 s5, s28, 8192
s_cbranch_scc0 label_7
s_load_dwordx8 [s48, s49, s50, s51, s52, s53, s54, s55], [s2, s3], 104
label_7:
s_and_b32 s5, s28, 512
s_cbranch_scc0 label_8
s_load_dwordx8 [s56, s57, s58, s59, s60, s61, s62, s63], [s2, s3], 136
s_load_dwordx4 [s64, s65, s66, s67], [s2, s3], 168
label_8:
s_and_b32 s5, s28, 1024
s_cbranch_scc0 label_9
s_load_dwordx4 [s68, s69, s70, s71], [s2, s3], 184
label_9:
s_and_b32 s5, s28, 81920
s_cbranch_scc0 label_10
s_load_dword s72, [s2, s3], 200
label_10:
s_and_b32 s5, s28, 65536
s_cbranch_scc0 label_11
s_load_dword s73, [s2, s3], 204
s_load_dwordx2 [s74, s75], [s2, s3], 208
label_11:
s_and_b32 s5, s28, 131072
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
s_mul_i32 s5, s38, s39
s_bitcmp1_b32 s28, 2
s_cselect_b32 s8, s26, s23
s_mul_i32 s8, s5, s8
s_mov_b32 s62, s39
s_bitcmp1_b32 s28, 2
s_cselect_b32 s61, s8, s5
s_cselect_b32 s60, s5, s8
label_18:
s_and_b32 s8, s72, -256
s_bitcmp1_b32 s28, 8
s_cselect_b32 s9, 1, 0
s_bitcmp1_b32 s28, 14
s_cselect_b32 s72, s72, s9
s_or_b32 s72, s72, s8
s_cmp_lt_u32 s60, s61
s_cselect_b32 s5, 0, 2147483648
s_or_b32 s28, s28, s5
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
v_readfirstlane_b32 s5, v2
s_mul_i32 s8, s5, s27
s_sub_u32 s4, s4, s8
s_mul_i32 s8, s69, s5
s_mul_hi_u32 s9, s69, s5
s_lshl_b64 [s8, s9], [s8, s9], 1
s_add_u32 s30, s30, s8
s_addc_u32 s31, s31, s9
s_mul_i32 s8, s70, s5
s_mul_hi_u32 s9, s70, s5
s_lshl_b64 [s8, s9], [s8, s9], 1
s_add_u32 s32, s32, s8
s_addc_u32 s33, s33, s9
s_mul_i32 s8, s71, s5
s_mul_hi_u32 s9, s71, s5
s_lshl_b64 [s8, s9], [s8, s9], 1
s_add_u32 s34, s34, s8
s_addc_u32 s35, s35, s9
s_mul_i32 s8, s26, s5
s_lshl_b32 s8, s8, 1
s_add_u32 s44, s44, s8
s_addc_u32 s45, s45, 0
label_20:
s_cmp_eq_u32 s27, 304
s_cbranch_scc0 label_21
s_and_b32 s5, s4, 7
s_lshr_b32 s8, s4, 3
s_mul_i32 s4, s5, 38
s_add_u32 s4, s4, s8
label_21:
s_add_u32 s9, s26, 31
s_lshr_b32 s9, s9, 5
v_ffbh_u32 v6, s9
v_lshlrev_b32 v7, v6, s9
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
v_readfirstlane_b32 s5, v2
v_readfirstlane_b32 s10, v4
s_cmp_ge_u32 s10, s9
s_cbranch_scc1 label_0
s_mul_i32 s27, s5, s9
s_mul_i32 s8, s10, s5
s_sub_u32 s8, s4, s8
s_mul_i32 s11, s10, 32
s_sub_u32 s26, s26, s11
s_min_u32 s26, s26, 32
s_lshl_b32 s11, s11, 1
s_mul_i32 s12, s11, s60
s_mul_hi_u32 s13, s11, s60
s_add_u32 s32, s32, s12
s_addc_u32 s33, s33, s13
s_mul_i32 s12, s11, s65
s_mul_hi_u32 s13, s11, s65
s_add_u32 s34, s34, s12
s_addc_u32 s35, s35, s13
s_add_u32 s44, s44, s11
s_addc_u32 s45, s45, 0
s_cmp_ge_u32 s1, 8
s_cbranch_scc1 label_5
label_4:
v_and_b32 v19, 255, v0
s_bitcmp1_b32 s28, 31
s_cselect_b32 s48, 2, 0
s_cselect_b32 s49, 0, 2
v_bfe_u32 v17, v19, 2, s48
v_bfe_u32 v20, v19, 6, s49
v_add_u32 v20, v17, v20
v_lshrrev_b32 v21, 2, v19
v_lshrrev_b32 v21, s48, v21
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
s_cmp_eq_u32 s1, 9
s_cselect_b32 s50, 1024, 0
s_cselect_b32 s51, 4, 0
s_cmp_ge_u32 s1, 10
s_cselect_b32 s50, 65536, s50
s_cselect_b32 s51, 65536, s51
s_and_b32 s53, s1, 3
s_mul_i32 s52, s53, 2048
v_and_b32 v17, 63, v0
v_lshl_add_u32 v5, v17, 4, s50
v_mov_b32 v7, s51
v_lshl_add_u32 v6, v17, 4, s52
s_mov_b32 s11, 0
s_mov_b32 s20, 0
s_lshl_b32 s48, s57, 1
s_lshl_b32 s49, s61, 1
s_and_b32 s50, s1, 3
s_mul_i32 s51, 1, s50
s_sub_u32 s9, s23, s51
s_addc_u32 s9, s9, 3
s_lshl_b32 s51, s48, 0
s_mul_i32 s51, s51, s50
s_add_u32 s30, s30, s51
s_addc_u32 s31, s31, 0
s_lshl_b32 s48, s48, 2
s_lshl_b32 s49, s49, 2
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
s_mov_b32 s10, 2147483648
s_cmp_ge_u32 s1, 4
s_cbranch_scc0 label_22
s_branch label_34
.align 32
label_50:
ds_read_b32 v16, v7 offset:55808
ds_read_b128 [v8, v9, v10, v11], v6 offset:45568
ds_read_b128 [v12, v13, v14, v15], v6 offset:46592
s_sub_u32 s11, s23, 1
s_add_u32 s36, s6, s29
s_addc_u32 s37, s7, 0
s_waitcnt lgkmcnt(2)
v_readfirstlane_b32 s21, v16
s_bitcmp1_b32 s21, 0
s_cmov_b32 s10, 0
s_add_u32 s20, s20, s21
s_mov_b32 s2, s9
s_mov_b64 [s12, s13], [s32, s33]
s_mov_b32 s14, s10
s_mov_b32 s3, s9
s_mov_b64 [s16, s17], [s32, s33]
s_mov_b32 s18, s10
s_setpc_b64 [s36, s37]
.align 32
label_51:
ds_read_b32 v16, v7 offset:55808
ds_read_b128 [v8, v9, v10, v11], v5 offset:53760
ds_read_b128 [v12, v13, v14, v15], v5 offset:54784
s_sub_u32 s11, s23, 1
s_add_u32 s36, s6, s29
s_addc_u32 s37, s7, 0
s_waitcnt lgkmcnt(2)
v_readfirstlane_b32 s21, v16
s_bitcmp1_b32 s21, 0
s_cmov_b32 s10, 0
s_add_u32 s20, s20, s21
s_mov_b32 s2, s9
s_mov_b64 [s12, s13], [s30, s31]
s_mov_b32 s14, s10
s_mov_b32 s3, s9
s_mov_b64 [s16, s17], [s30, s31]
s_mov_b32 s18, s10
s_setpc_b64 [s36, s37]
.align 32
label_22:
s_lshl_b32 s20, s20, 2
s_setprio 1
s_waitcnt vmcnt(24)
s_waitcnt lgkmcnt(0)
v_pk_fma_f16 v34, v86, -1.0, v34 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v35, v87, -1.0, v35 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v17, v17, v129 op_sel:[0, 0, 0]
v_pack_b32_f16 v106, v106, v130 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v18, v19, v20, v21], [v90, v91], [v34, v35], [v18, v19, v20, v21]
v_pk_fma_f16 v88, v68, -1.0, v88 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v89, v69, -1.0, v89 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v109, v109, v133 op_sel:[0, 0, 0]
v_pack_b32_f16 v110, v110, v134 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v70, v71, v72, v73], [v102, v103], [v88, v89], [v70, v71, v72, v73]
v_pk_fma_f16 v86, v68, -1.0, v86 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v87, v69, -1.0, v87 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v112, v112, v136 op_sel:[0, 0, 0]
v_pack_b32_f16 v111, v111, v135 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v52, v53, v54, v55], [v98, v99], [v86, v87], [v52, v53, v54, v55]
v_pk_fma_f16 v68, v68, 2.0, v86 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v69, v69, 2.0, v87 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_mov_b32 v154, v17
v_mov_b32 v155, v106
v_mfma_f32_16x16x16_f16 [v36, v37, v38, v39], [v94, v95], [v68, v69], [v36, v37, v38, v39]
v_mov_b32 v156, v112
v_mov_b32 v157, v111
v_pk_add_f16 v112, v156, v154 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v111, v157, v155 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
s_barrier
s_setprio 2
s_cmp_lt_u32 s11, 15
s_cbranch_scc1 label_53
label_52:
ds_read_b64 [v92, v93], v2 offset:39296
buffer_load_short_d16 v17, v12, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v154, v155] offset:33024
buffer_load_short_d16 v106, v13, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v96, v97], v2 offset:39304
buffer_load_short_d16 v107, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s49
s_addc_u32 s17, s17, 0
ds_write_b64 v4, [v156, v157] offset:34176
buffer_load_short_d16 v129, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v104, v105], v2 offset:39312
buffer_load_short_d16 v130, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16 v131, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s49
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_23
s_mov_b32 s29, label_23 - label_1
s_branch label_46
.align 32
label_53:
v_sub_co_u32 v15, [vcc_lo, vcc_hi], v15, 4
v_cndmask_b32 v12, v12, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v13, v13, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v14, v14, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v92, v93], v2 offset:39296
buffer_load_short_d16 v17, v12, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v154, v155] offset:33024
buffer_load_short_d16 v106, v13, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v96, v97], v2 offset:39304
buffer_load_short_d16 v107, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s49
s_addc_u32 s17, s17, 0
v_sub_co_u32 v15, [vcc_lo, vcc_hi], v15, 4
v_cndmask_b32 v12, v12, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v13, v13, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v14, v14, -1, [vcc_lo, vcc_hi]
ds_write_b64 v4, [v156, v157] offset:34176
buffer_load_short_d16 v129, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v104, v105], v2 offset:39312
buffer_load_short_d16 v130, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16 v131, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s49
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_23
s_mov_b32 s29, label_23 - label_1
s_branch label_46
.align 32
label_23:
s_setprio 1
s_waitcnt lgkmcnt(0)
v_pk_add_f16 v100, v92, v104 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v101, v93, v105 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mfma_f32_16x16x16_f16 [v26, v27, v28, v29], [v92, v93], [v34, v35], [v26, v27, v28, v29]
v_pk_mul_f16 v100, v100, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_pk_mul_f16 v101, v101, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_pk_add_f16 v154, v112, v109 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v155, v111, v110 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mfma_f32_16x16x16_f16 [v78, v79, v80, v81], [v104, v105], [v88, v89], [v78, v79, v80, v81]
v_pk_fma_f16 v100, v96, -0.5, v100 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v101, v97, -0.5, v101 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_mul_f16 v154, v154, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_pk_mul_f16 v155, v155, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_mfma_f32_16x16x16_f16 [v60, v61, v62, v63], [v100, v101], [v86, v87], [v60, v61, v62, v63]
v_pk_fma_f16 v96, v96, 1.0, v100 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v97, v97, 1.0, v101 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v156, v154, -1.0, v112 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v157, v155, -1.0, v111 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_mfma_f32_16x16x16_f16 [v44, v45, v46, v47], [v96, v97], [v68, v69], [v44, v45, v46, v47]
s_barrier
s_setprio 2
s_cmp_lt_u32 s11, 15
s_cbranch_scc1 label_55
label_54:
ds_read_b64 [v34, v35], v1 offset:24768
buffer_load_short_d16 v109, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v68, v69], v1 offset:28928
buffer_load_short_d16 v110, v13, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v154, v155] offset:33408
buffer_load_short_d16 v111, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s49
s_addc_u32 s17, s17, 0
ds_read_b64 [v86, v87], v1 offset:24776
buffer_load_short_d16 v133, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v88, v89], v1 offset:28936
buffer_load_short_d16 v134, v13, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v156, v157] offset:33792
buffer_load_short_d16 v135, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s49
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_24
s_mov_b32 s29, label_24 - label_1
s_branch label_47
.align 32
label_55:
v_sub_co_u32 v15, [vcc_lo, vcc_hi], v15, 4
v_cndmask_b32 v12, v12, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v13, v13, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v14, v14, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v34, v35], v1 offset:24768
buffer_load_short_d16 v109, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v68, v69], v1 offset:28928
buffer_load_short_d16 v110, v13, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v154, v155] offset:33408
buffer_load_short_d16 v111, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s49
s_addc_u32 s17, s17, 0
v_sub_co_u32 v15, [vcc_lo, vcc_hi], v15, 4
v_cndmask_b32 v12, v12, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v13, v13, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v14, v14, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v86, v87], v1 offset:24776
buffer_load_short_d16 v133, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v88, v89], v1 offset:28936
buffer_load_short_d16 v134, v13, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v156, v157] offset:33792
buffer_load_short_d16 v135, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s49
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_24
s_mov_b32 s29, label_24 - label_1
s_branch label_47
.align 32
label_24:
s_mov_b32 s29, label_56 - label_1
s_sub_u32 s11, s11, 16
s_cbranch_scc1 label_50
label_56:
s_setprio 1
s_waitcnt vmcnt(24)
s_waitcnt lgkmcnt(0)
v_pk_fma_f16 v34, v86, -1.0, v34 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v35, v87, -1.0, v35 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v113, v113, v137 op_sel:[0, 0, 0]
v_pack_b32_f16 v114, v114, v138 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v22, v23, v24, v25], [v90, v91], [v34, v35], [v22, v23, v24, v25]
v_pk_fma_f16 v88, v68, -1.0, v88 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v89, v69, -1.0, v89 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v117, v117, v141 op_sel:[0, 0, 0]
v_pack_b32_f16 v118, v118, v142 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v74, v75, v76, v77], [v102, v103], [v88, v89], [v74, v75, v76, v77]
v_pk_fma_f16 v86, v68, -1.0, v86 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v87, v69, -1.0, v87 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v120, v120, v144 op_sel:[0, 0, 0]
v_pack_b32_f16 v119, v119, v143 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v56, v57, v58, v59], [v98, v99], [v86, v87], [v56, v57, v58, v59]
v_pk_fma_f16 v68, v68, 2.0, v86 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v69, v69, 2.0, v87 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_mov_b32 v154, v113
v_mov_b32 v155, v114
v_mfma_f32_16x16x16_f16 [v40, v41, v42, v43], [v94, v95], [v68, v69], [v40, v41, v42, v43]
v_mov_b32 v156, v120
v_mov_b32 v157, v119
v_pk_add_f16 v120, v156, v154 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v119, v157, v155 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
s_barrier
s_setprio 2
s_cmp_lt_u32 s11, 15
s_cbranch_scc1 label_58
label_57:
ds_read_b64 [v90, v91], v2 offset:33024
buffer_load_short_d16 v113, v8, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v154, v155] offset:39296
buffer_load_short_d16 v114, v9, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v94, v95], v2 offset:33032
buffer_load_short_d16 v115, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s49
s_addc_u32 s13, s13, 0
ds_write_b64 v4, [v156, v157] offset:40448
buffer_load_short_d16 v137, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v102, v103], v2 offset:33040
buffer_load_short_d16 v138, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16 v139, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s49
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_25
s_mov_b32 s29, label_25 - label_1
s_branch label_48
.align 32
label_58:
v_sub_co_u32 v11, [vcc_lo, vcc_hi], v11, 4
v_cndmask_b32 v8, v8, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v9, v9, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v10, v10, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v90, v91], v2 offset:33024
buffer_load_short_d16 v113, v8, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v154, v155] offset:39296
buffer_load_short_d16 v114, v9, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v94, v95], v2 offset:33032
buffer_load_short_d16 v115, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s49
s_addc_u32 s13, s13, 0
v_sub_co_u32 v11, [vcc_lo, vcc_hi], v11, 4
v_cndmask_b32 v8, v8, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v9, v9, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v10, v10, -1, [vcc_lo, vcc_hi]
ds_write_b64 v4, [v156, v157] offset:40448
buffer_load_short_d16 v137, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v102, v103], v2 offset:33040
buffer_load_short_d16 v138, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16 v139, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s49
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_25
s_mov_b32 s29, label_25 - label_1
s_branch label_48
.align 32
label_25:
s_setprio 1
s_waitcnt lgkmcnt(0)
v_pk_add_f16 v98, v90, v102 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v99, v91, v103 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mfma_f32_16x16x16_f16 [v30, v31, v32, v33], [v92, v93], [v34, v35], [v30, v31, v32, v33]
v_pk_mul_f16 v98, v98, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_pk_mul_f16 v99, v99, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_pk_add_f16 v154, v120, v117 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v155, v119, v118 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mfma_f32_16x16x16_f16 [v82, v83, v84, v85], [v104, v105], [v88, v89], [v82, v83, v84, v85]
v_pk_fma_f16 v98, v94, -0.5, v98 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v99, v95, -0.5, v99 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_mul_f16 v154, v154, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_pk_mul_f16 v155, v155, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_mfma_f32_16x16x16_f16 [v64, v65, v66, v67], [v100, v101], [v86, v87], [v64, v65, v66, v67]
v_pk_fma_f16 v94, v94, 1.0, v98 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v95, v95, 1.0, v99 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v156, v154, -1.0, v120 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v157, v155, -1.0, v119 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_mfma_f32_16x16x16_f16 [v48, v49, v50, v51], [v96, v97], [v68, v69], [v48, v49, v50, v51]
s_barrier
s_setprio 2
s_cmp_lt_u32 s11, 15
s_cbranch_scc1 label_60
label_59:
ds_read_b64 [v34, v35], v1 offset:16512
buffer_load_short_d16 v117, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v68, v69], v1 offset:20672
buffer_load_short_d16 v118, v9, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v154, v155] offset:39680
buffer_load_short_d16 v119, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s49
s_addc_u32 s13, s13, 0
ds_read_b64 [v86, v87], v1 offset:16520
buffer_load_short_d16 v141, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v88, v89], v1 offset:20680
buffer_load_short_d16 v142, v9, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v156, v157] offset:40064
buffer_load_short_d16 v143, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s49
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_26
s_mov_b32 s29, label_26 - label_1
s_branch label_49
.align 32
label_60:
v_sub_co_u32 v11, [vcc_lo, vcc_hi], v11, 4
v_cndmask_b32 v8, v8, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v9, v9, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v10, v10, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v34, v35], v1 offset:16512
buffer_load_short_d16 v117, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v68, v69], v1 offset:20672
buffer_load_short_d16 v118, v9, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v154, v155] offset:39680
buffer_load_short_d16 v119, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s49
s_addc_u32 s13, s13, 0
v_sub_co_u32 v11, [vcc_lo, vcc_hi], v11, 4
v_cndmask_b32 v8, v8, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v9, v9, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v10, v10, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v86, v87], v1 offset:16520
buffer_load_short_d16 v141, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v88, v89], v1 offset:20680
buffer_load_short_d16 v142, v9, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v156, v157] offset:40064
buffer_load_short_d16 v143, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s49
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_26
s_mov_b32 s29, label_26 - label_1
s_branch label_49
.align 32
label_26:
s_lshl_b32 s20, s20, 2
s_setprio 1
s_waitcnt vmcnt(24)
s_waitcnt lgkmcnt(0)
v_pk_fma_f16 v34, v86, -1.0, v34 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v35, v87, -1.0, v35 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v122, v122, v146 op_sel:[0, 0, 0]
v_pack_b32_f16 v124, v124, v148 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v18, v19, v20, v21], [v90, v91], [v34, v35], [v18, v19, v20, v21]
v_pk_fma_f16 v88, v68, -1.0, v88 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v89, v69, -1.0, v89 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v126, v126, v150 op_sel:[0, 0, 0]
v_pack_b32_f16 v128, v128, v152 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v70, v71, v72, v73], [v102, v103], [v88, v89], [v70, v71, v72, v73]
v_pk_fma_f16 v86, v68, -1.0, v86 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v87, v69, -1.0, v87 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v127, v127, v151 op_sel:[0, 0, 0]
v_pack_b32_f16 v125, v125, v149 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v52, v53, v54, v55], [v98, v99], [v86, v87], [v52, v53, v54, v55]
v_pk_fma_f16 v68, v68, 2.0, v86 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v69, v69, 2.0, v87 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_mov_b32 v154, v122
v_mov_b32 v155, v124
v_mfma_f32_16x16x16_f16 [v36, v37, v38, v39], [v94, v95], [v68, v69], [v36, v37, v38, v39]
v_mov_b32 v156, v127
v_mov_b32 v157, v125
v_pk_add_f16 v127, v156, v154 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v125, v157, v155 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
s_barrier
s_setprio 2
s_cmp_lt_u32 s11, 15
s_cbranch_scc1 label_62
label_61:
ds_read_b64 [v92, v93], v2 offset:39296
buffer_load_short_d16 v122, v12, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v154, v155] offset:33024
buffer_load_short_d16 v124, v13, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v96, v97], v2 offset:39304
buffer_load_short_d16 v121, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s49
s_addc_u32 s17, s17, 0
ds_write_b64 v4, [v156, v157] offset:34176
buffer_load_short_d16 v146, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v104, v105], v2 offset:39312
buffer_load_short_d16 v148, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16 v145, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s49
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_27
s_mov_b32 s29, label_27 - label_1
s_branch label_46
.align 32
label_62:
v_sub_co_u32 v15, [vcc_lo, vcc_hi], v15, 4
v_cndmask_b32 v12, v12, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v13, v13, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v14, v14, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v92, v93], v2 offset:39296
buffer_load_short_d16 v122, v12, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v154, v155] offset:33024
buffer_load_short_d16 v124, v13, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v96, v97], v2 offset:39304
buffer_load_short_d16 v121, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s49
s_addc_u32 s17, s17, 0
v_sub_co_u32 v15, [vcc_lo, vcc_hi], v15, 4
v_cndmask_b32 v12, v12, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v13, v13, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v14, v14, -1, [vcc_lo, vcc_hi]
ds_write_b64 v4, [v156, v157] offset:34176
buffer_load_short_d16 v146, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v104, v105], v2 offset:39312
buffer_load_short_d16 v148, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16 v145, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s49
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_27
s_mov_b32 s29, label_27 - label_1
s_branch label_46
.align 32
label_27:
s_setprio 1
s_waitcnt lgkmcnt(0)
v_pk_add_f16 v100, v92, v104 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v101, v93, v105 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mfma_f32_16x16x16_f16 [v26, v27, v28, v29], [v92, v93], [v34, v35], [v26, v27, v28, v29]
v_pk_mul_f16 v100, v100, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_pk_mul_f16 v101, v101, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_pk_add_f16 v154, v127, v126 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v155, v125, v128 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mfma_f32_16x16x16_f16 [v78, v79, v80, v81], [v104, v105], [v88, v89], [v78, v79, v80, v81]
v_pk_fma_f16 v100, v96, -0.5, v100 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v101, v97, -0.5, v101 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_mul_f16 v154, v154, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_pk_mul_f16 v155, v155, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_mfma_f32_16x16x16_f16 [v60, v61, v62, v63], [v100, v101], [v86, v87], [v60, v61, v62, v63]
v_pk_fma_f16 v96, v96, 1.0, v100 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v97, v97, 1.0, v101 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v156, v154, -1.0, v127 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v157, v155, -1.0, v125 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_mfma_f32_16x16x16_f16 [v44, v45, v46, v47], [v96, v97], [v68, v69], [v44, v45, v46, v47]
s_barrier
s_setprio 2
s_cmp_lt_u32 s11, 15
s_cbranch_scc1 label_64
label_63:
ds_read_b64 [v34, v35], v1 offset:24768
buffer_load_short_d16 v126, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v68, v69], v1 offset:28928
buffer_load_short_d16 v128, v13, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v154, v155] offset:33408
buffer_load_short_d16 v125, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s49
s_addc_u32 s17, s17, 0
ds_read_b64 [v86, v87], v1 offset:24776
buffer_load_short_d16 v150, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v88, v89], v1 offset:28936
buffer_load_short_d16 v152, v13, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v156, v157] offset:33792
buffer_load_short_d16 v149, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s49
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_28
s_mov_b32 s29, label_28 - label_1
s_branch label_47
.align 32
label_64:
v_sub_co_u32 v15, [vcc_lo, vcc_hi], v15, 4
v_cndmask_b32 v12, v12, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v13, v13, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v14, v14, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v34, v35], v1 offset:24768
buffer_load_short_d16 v126, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v68, v69], v1 offset:28928
buffer_load_short_d16 v128, v13, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v154, v155] offset:33408
buffer_load_short_d16 v125, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s49
s_addc_u32 s17, s17, 0
v_sub_co_u32 v15, [vcc_lo, vcc_hi], v15, 4
v_cndmask_b32 v12, v12, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v13, v13, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v14, v14, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v86, v87], v1 offset:24776
buffer_load_short_d16 v150, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v88, v89], v1 offset:28936
buffer_load_short_d16 v152, v13, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v156, v157] offset:33792
buffer_load_short_d16 v149, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s49
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_28
s_mov_b32 s29, label_28 - label_1
s_branch label_47
.align 32
label_28:
s_mov_b32 s29, label_65 - label_1
s_sub_u32 s11, s11, 16
s_cbranch_scc1 label_50
label_65:
s_setprio 1
s_waitcnt vmcnt(24)
s_waitcnt lgkmcnt(0)
v_pk_fma_f16 v34, v86, -1.0, v34 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v35, v87, -1.0, v35 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v17, v17, v129 op_sel:[0, 0, 0]
v_pack_b32_f16 v109, v109, v133 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v22, v23, v24, v25], [v90, v91], [v34, v35], [v22, v23, v24, v25]
v_pk_fma_f16 v88, v68, -1.0, v88 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v89, v69, -1.0, v89 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v106, v106, v130 op_sel:[0, 0, 0]
v_pack_b32_f16 v110, v110, v134 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v74, v75, v76, v77], [v102, v103], [v88, v89], [v74, v75, v76, v77]
v_pk_fma_f16 v86, v68, -1.0, v86 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v87, v69, -1.0, v87 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v107, v107, v131 op_sel:[0, 0, 0]
v_pack_b32_f16 v111, v111, v135 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v56, v57, v58, v59], [v98, v99], [v86, v87], [v56, v57, v58, v59]
v_pk_fma_f16 v68, v68, 2.0, v86 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v69, v69, 2.0, v87 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_mov_b32 v154, v17
v_mov_b32 v155, v109
v_mfma_f32_16x16x16_f16 [v40, v41, v42, v43], [v94, v95], [v68, v69], [v40, v41, v42, v43]
v_mov_b32 v156, v107
v_mov_b32 v157, v111
v_pk_add_f16 v107, v156, v154 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v111, v157, v155 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
s_barrier
s_setprio 2
s_cmp_lt_u32 s11, 15
s_cbranch_scc1 label_67
label_66:
ds_read_b64 [v90, v91], v2 offset:33024
buffer_load_short_d16 v17, v8, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v154, v155] offset:39296
buffer_load_short_d16 v109, v9, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v94, v95], v2 offset:33032
buffer_load_short_d16 v112, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s49
s_addc_u32 s13, s13, 0
ds_write_b64 v4, [v156, v157] offset:40448
buffer_load_short_d16 v129, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v102, v103], v2 offset:33040
buffer_load_short_d16 v133, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16 v136, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s49
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_29
s_mov_b32 s29, label_29 - label_1
s_branch label_48
.align 32
label_67:
v_sub_co_u32 v11, [vcc_lo, vcc_hi], v11, 4
v_cndmask_b32 v8, v8, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v9, v9, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v10, v10, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v90, v91], v2 offset:33024
buffer_load_short_d16 v17, v8, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v154, v155] offset:39296
buffer_load_short_d16 v109, v9, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v94, v95], v2 offset:33032
buffer_load_short_d16 v112, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s49
s_addc_u32 s13, s13, 0
v_sub_co_u32 v11, [vcc_lo, vcc_hi], v11, 4
v_cndmask_b32 v8, v8, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v9, v9, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v10, v10, -1, [vcc_lo, vcc_hi]
ds_write_b64 v4, [v156, v157] offset:40448
buffer_load_short_d16 v129, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v102, v103], v2 offset:33040
buffer_load_short_d16 v133, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16 v136, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s49
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_29
s_mov_b32 s29, label_29 - label_1
s_branch label_48
.align 32
label_29:
s_setprio 1
s_waitcnt lgkmcnt(0)
v_pk_add_f16 v98, v90, v102 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v99, v91, v103 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mfma_f32_16x16x16_f16 [v30, v31, v32, v33], [v92, v93], [v34, v35], [v30, v31, v32, v33]
v_pk_mul_f16 v98, v98, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_pk_mul_f16 v99, v99, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_pk_add_f16 v154, v107, v106 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v155, v111, v110 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mfma_f32_16x16x16_f16 [v82, v83, v84, v85], [v104, v105], [v88, v89], [v82, v83, v84, v85]
v_pk_fma_f16 v98, v94, -0.5, v98 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v99, v95, -0.5, v99 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_mul_f16 v154, v154, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_pk_mul_f16 v155, v155, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_mfma_f32_16x16x16_f16 [v64, v65, v66, v67], [v100, v101], [v86, v87], [v64, v65, v66, v67]
v_pk_fma_f16 v94, v94, 1.0, v98 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v95, v95, 1.0, v99 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v156, v154, -1.0, v107 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v157, v155, -1.0, v111 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_mfma_f32_16x16x16_f16 [v48, v49, v50, v51], [v96, v97], [v68, v69], [v48, v49, v50, v51]
s_barrier
s_setprio 2
s_cmp_lt_u32 s11, 15
s_cbranch_scc1 label_69
label_68:
ds_read_b64 [v34, v35], v1 offset:16512
buffer_load_short_d16 v106, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v68, v69], v1 offset:20672
buffer_load_short_d16 v110, v9, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v154, v155] offset:39680
buffer_load_short_d16 v111, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s49
s_addc_u32 s13, s13, 0
ds_read_b64 [v86, v87], v1 offset:16520
buffer_load_short_d16 v130, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v88, v89], v1 offset:20680
buffer_load_short_d16 v134, v9, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v156, v157] offset:40064
buffer_load_short_d16 v135, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s49
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_30
s_mov_b32 s29, label_30 - label_1
s_branch label_49
.align 32
label_69:
v_sub_co_u32 v11, [vcc_lo, vcc_hi], v11, 4
v_cndmask_b32 v8, v8, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v9, v9, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v10, v10, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v34, v35], v1 offset:16512
buffer_load_short_d16 v106, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v68, v69], v1 offset:20672
buffer_load_short_d16 v110, v9, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v154, v155] offset:39680
buffer_load_short_d16 v111, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s49
s_addc_u32 s13, s13, 0
v_sub_co_u32 v11, [vcc_lo, vcc_hi], v11, 4
v_cndmask_b32 v8, v8, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v9, v9, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v10, v10, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v86, v87], v1 offset:16520
buffer_load_short_d16 v130, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v88, v89], v1 offset:20680
buffer_load_short_d16 v134, v9, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v156, v157] offset:40064
buffer_load_short_d16 v135, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s49
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_30
s_mov_b32 s29, label_30 - label_1
s_branch label_49
.align 32
label_30:
s_lshl_b32 s20, s20, 2
s_setprio 1
s_waitcnt vmcnt(24)
s_waitcnt lgkmcnt(0)
v_pk_fma_f16 v34, v86, -1.0, v34 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v35, v87, -1.0, v35 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v113, v113, v137 op_sel:[0, 0, 0]
v_pack_b32_f16 v117, v117, v141 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v18, v19, v20, v21], [v90, v91], [v34, v35], [v18, v19, v20, v21]
v_pk_fma_f16 v88, v68, -1.0, v88 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v89, v69, -1.0, v89 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v114, v114, v138 op_sel:[0, 0, 0]
v_pack_b32_f16 v118, v118, v142 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v70, v71, v72, v73], [v102, v103], [v88, v89], [v70, v71, v72, v73]
v_pk_fma_f16 v86, v68, -1.0, v86 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v87, v69, -1.0, v87 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v115, v115, v139 op_sel:[0, 0, 0]
v_pack_b32_f16 v119, v119, v143 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v52, v53, v54, v55], [v98, v99], [v86, v87], [v52, v53, v54, v55]
v_pk_fma_f16 v68, v68, 2.0, v86 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v69, v69, 2.0, v87 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_mov_b32 v154, v113
v_mov_b32 v155, v117
v_mfma_f32_16x16x16_f16 [v36, v37, v38, v39], [v94, v95], [v68, v69], [v36, v37, v38, v39]
v_mov_b32 v156, v115
v_mov_b32 v157, v119
v_pk_add_f16 v115, v156, v154 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v119, v157, v155 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
s_barrier
s_setprio 2
s_cmp_lt_u32 s11, 15
s_cbranch_scc1 label_71
label_70:
ds_read_b64 [v92, v93], v2 offset:39296
buffer_load_short_d16 v113, v12, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v154, v155] offset:33024
buffer_load_short_d16 v117, v13, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v96, v97], v2 offset:39304
buffer_load_short_d16 v120, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s49
s_addc_u32 s17, s17, 0
ds_write_b64 v4, [v156, v157] offset:34176
buffer_load_short_d16 v137, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v104, v105], v2 offset:39312
buffer_load_short_d16 v141, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16 v144, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s49
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_31
s_mov_b32 s29, label_31 - label_1
s_branch label_46
.align 32
label_71:
v_sub_co_u32 v15, [vcc_lo, vcc_hi], v15, 4
v_cndmask_b32 v12, v12, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v13, v13, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v14, v14, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v92, v93], v2 offset:39296
buffer_load_short_d16 v113, v12, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v154, v155] offset:33024
buffer_load_short_d16 v117, v13, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v96, v97], v2 offset:39304
buffer_load_short_d16 v120, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s49
s_addc_u32 s17, s17, 0
v_sub_co_u32 v15, [vcc_lo, vcc_hi], v15, 4
v_cndmask_b32 v12, v12, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v13, v13, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v14, v14, -1, [vcc_lo, vcc_hi]
ds_write_b64 v4, [v156, v157] offset:34176
buffer_load_short_d16 v137, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v104, v105], v2 offset:39312
buffer_load_short_d16 v141, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16 v144, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s49
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_31
s_mov_b32 s29, label_31 - label_1
s_branch label_46
.align 32
label_31:
s_setprio 1
s_waitcnt lgkmcnt(0)
v_pk_add_f16 v100, v92, v104 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v101, v93, v105 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mfma_f32_16x16x16_f16 [v26, v27, v28, v29], [v92, v93], [v34, v35], [v26, v27, v28, v29]
v_pk_mul_f16 v100, v100, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_pk_mul_f16 v101, v101, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_pk_add_f16 v154, v115, v114 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v155, v119, v118 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mfma_f32_16x16x16_f16 [v78, v79, v80, v81], [v104, v105], [v88, v89], [v78, v79, v80, v81]
v_pk_fma_f16 v100, v96, -0.5, v100 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v101, v97, -0.5, v101 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_mul_f16 v154, v154, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_pk_mul_f16 v155, v155, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_mfma_f32_16x16x16_f16 [v60, v61, v62, v63], [v100, v101], [v86, v87], [v60, v61, v62, v63]
v_pk_fma_f16 v96, v96, 1.0, v100 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v97, v97, 1.0, v101 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v156, v154, -1.0, v115 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v157, v155, -1.0, v119 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_mfma_f32_16x16x16_f16 [v44, v45, v46, v47], [v96, v97], [v68, v69], [v44, v45, v46, v47]
s_barrier
s_setprio 2
s_cmp_lt_u32 s11, 15
s_cbranch_scc1 label_73
label_72:
ds_read_b64 [v34, v35], v1 offset:24768
buffer_load_short_d16 v114, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v68, v69], v1 offset:28928
buffer_load_short_d16 v118, v13, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v154, v155] offset:33408
buffer_load_short_d16 v119, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s49
s_addc_u32 s17, s17, 0
ds_read_b64 [v86, v87], v1 offset:24776
buffer_load_short_d16 v138, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v88, v89], v1 offset:28936
buffer_load_short_d16 v142, v13, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v156, v157] offset:33792
buffer_load_short_d16 v143, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s49
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_32
s_mov_b32 s29, label_32 - label_1
s_branch label_47
.align 32
label_73:
v_sub_co_u32 v15, [vcc_lo, vcc_hi], v15, 4
v_cndmask_b32 v12, v12, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v13, v13, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v14, v14, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v34, v35], v1 offset:24768
buffer_load_short_d16 v114, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v68, v69], v1 offset:28928
buffer_load_short_d16 v118, v13, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v154, v155] offset:33408
buffer_load_short_d16 v119, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s49
s_addc_u32 s17, s17, 0
v_sub_co_u32 v15, [vcc_lo, vcc_hi], v15, 4
v_cndmask_b32 v12, v12, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v13, v13, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v14, v14, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v86, v87], v1 offset:24776
buffer_load_short_d16 v138, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v88, v89], v1 offset:28936
buffer_load_short_d16 v142, v13, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v156, v157] offset:33792
buffer_load_short_d16 v143, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s49
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_32
s_mov_b32 s29, label_32 - label_1
s_branch label_47
.align 32
label_32:
s_mov_b32 s29, label_74 - label_1
s_sub_u32 s11, s11, 16
s_cbranch_scc1 label_50
label_74:
s_setprio 1
s_waitcnt vmcnt(24)
s_waitcnt lgkmcnt(0)
v_pk_fma_f16 v34, v86, -1.0, v34 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v35, v87, -1.0, v35 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v122, v122, v146 op_sel:[0, 0, 0]
v_pack_b32_f16 v126, v126, v150 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v22, v23, v24, v25], [v90, v91], [v34, v35], [v22, v23, v24, v25]
v_pk_fma_f16 v88, v68, -1.0, v88 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v89, v69, -1.0, v89 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v124, v124, v148 op_sel:[0, 0, 0]
v_pack_b32_f16 v128, v128, v152 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v74, v75, v76, v77], [v102, v103], [v88, v89], [v74, v75, v76, v77]
v_pk_fma_f16 v86, v68, -1.0, v86 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v87, v69, -1.0, v87 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v121, v121, v145 op_sel:[0, 0, 0]
v_pack_b32_f16 v125, v125, v149 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v56, v57, v58, v59], [v98, v99], [v86, v87], [v56, v57, v58, v59]
v_pk_fma_f16 v68, v68, 2.0, v86 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v69, v69, 2.0, v87 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_mov_b32 v154, v122
v_mov_b32 v155, v126
v_mfma_f32_16x16x16_f16 [v40, v41, v42, v43], [v94, v95], [v68, v69], [v40, v41, v42, v43]
v_mov_b32 v156, v121
v_mov_b32 v157, v125
v_pk_add_f16 v121, v156, v154 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v125, v157, v155 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
s_barrier
s_setprio 2
s_cmp_lt_u32 s11, 15
s_cbranch_scc1 label_76
label_75:
ds_read_b64 [v90, v91], v2 offset:33024
buffer_load_short_d16 v122, v8, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v154, v155] offset:39296
buffer_load_short_d16 v126, v9, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v94, v95], v2 offset:33032
buffer_load_short_d16 v127, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s49
s_addc_u32 s13, s13, 0
ds_write_b64 v4, [v156, v157] offset:40448
buffer_load_short_d16 v146, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v102, v103], v2 offset:33040
buffer_load_short_d16 v150, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16 v151, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s49
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_33
s_mov_b32 s29, label_33 - label_1
s_branch label_48
.align 32
label_76:
v_sub_co_u32 v11, [vcc_lo, vcc_hi], v11, 4
v_cndmask_b32 v8, v8, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v9, v9, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v10, v10, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v90, v91], v2 offset:33024
buffer_load_short_d16 v122, v8, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v154, v155] offset:39296
buffer_load_short_d16 v126, v9, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v94, v95], v2 offset:33032
buffer_load_short_d16 v127, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s49
s_addc_u32 s13, s13, 0
v_sub_co_u32 v11, [vcc_lo, vcc_hi], v11, 4
v_cndmask_b32 v8, v8, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v9, v9, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v10, v10, -1, [vcc_lo, vcc_hi]
ds_write_b64 v4, [v156, v157] offset:40448
buffer_load_short_d16 v146, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v102, v103], v2 offset:33040
buffer_load_short_d16 v150, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16 v151, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s49
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_33
s_mov_b32 s29, label_33 - label_1
s_branch label_48
.align 32
label_33:
s_setprio 1
s_waitcnt lgkmcnt(0)
v_pk_add_f16 v98, v90, v102 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v99, v91, v103 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mfma_f32_16x16x16_f16 [v30, v31, v32, v33], [v92, v93], [v34, v35], [v30, v31, v32, v33]
v_pk_mul_f16 v98, v98, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_pk_mul_f16 v99, v99, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_pk_add_f16 v154, v121, v124 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v155, v125, v128 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mfma_f32_16x16x16_f16 [v82, v83, v84, v85], [v104, v105], [v88, v89], [v82, v83, v84, v85]
v_pk_fma_f16 v98, v94, -0.5, v98 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v99, v95, -0.5, v99 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_mul_f16 v154, v154, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_pk_mul_f16 v155, v155, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_mfma_f32_16x16x16_f16 [v64, v65, v66, v67], [v100, v101], [v86, v87], [v64, v65, v66, v67]
v_pk_fma_f16 v94, v94, 1.0, v98 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v95, v95, 1.0, v99 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v156, v154, -1.0, v121 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v157, v155, -1.0, v125 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_mfma_f32_16x16x16_f16 [v48, v49, v50, v51], [v96, v97], [v68, v69], [v48, v49, v50, v51]
s_barrier
s_setprio 2
s_cmp_lt_u32 s11, 15
s_cbranch_scc1 label_78
label_77:
ds_read_b64 [v34, v35], v1 offset:16512
buffer_load_short_d16 v124, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v68, v69], v1 offset:20672
buffer_load_short_d16 v128, v9, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v154, v155] offset:39680
buffer_load_short_d16 v125, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s49
s_addc_u32 s13, s13, 0
ds_read_b64 [v86, v87], v1 offset:16520
buffer_load_short_d16 v148, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v88, v89], v1 offset:20680
buffer_load_short_d16 v152, v9, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v156, v157] offset:40064
buffer_load_short_d16 v149, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s49
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_22
s_mov_b32 s29, label_22 - label_1
s_branch label_49
.align 32
label_78:
v_sub_co_u32 v11, [vcc_lo, vcc_hi], v11, 4
v_cndmask_b32 v8, v8, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v9, v9, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v10, v10, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v34, v35], v1 offset:16512
buffer_load_short_d16 v124, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v68, v69], v1 offset:20672
buffer_load_short_d16 v128, v9, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v154, v155] offset:39680
buffer_load_short_d16 v125, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s49
s_addc_u32 s13, s13, 0
v_sub_co_u32 v11, [vcc_lo, vcc_hi], v11, 4
v_cndmask_b32 v8, v8, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v9, v9, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v10, v10, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v86, v87], v1 offset:16520
buffer_load_short_d16 v148, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v88, v89], v1 offset:20680
buffer_load_short_d16 v152, v9, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v156, v157] offset:40064
buffer_load_short_d16 v149, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s49
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_22
s_mov_b32 s29, label_22 - label_1
s_branch label_49
.align 32
label_34:
s_lshl_b32 s20, s20, 2
s_setprio 1
s_waitcnt vmcnt(32)
s_waitcnt lgkmcnt(0)
s_barrier
v_pk_fma_f16 v34, v86, -1.0, v34 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v35, v87, -1.0, v35 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v17, v17, v129 op_sel:[0, 0, 0]
v_pack_b32_f16 v106, v106, v130 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v18, v19, v20, v21], [v90, v91], [v34, v35], [v18, v19, v20, v21]
v_pk_fma_f16 v88, v68, -1.0, v88 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v89, v69, -1.0, v89 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v109, v109, v133 op_sel:[0, 0, 0]
v_pack_b32_f16 v110, v110, v134 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v70, v71, v72, v73], [v102, v103], [v88, v89], [v70, v71, v72, v73]
v_pk_fma_f16 v86, v68, -1.0, v86 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v87, v69, -1.0, v87 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v108, v108, v132 op_sel:[0, 0, 0]
v_pack_b32_f16 v107, v107, v131 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v52, v53, v54, v55], [v98, v99], [v86, v87], [v52, v53, v54, v55]
v_pk_fma_f16 v68, v68, 2.0, v86 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v69, v69, 2.0, v87 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v112, v112, v136 op_sel:[0, 0, 0]
v_pack_b32_f16 v111, v111, v135 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v36, v37, v38, v39], [v94, v95], [v68, v69], [v36, v37, v38, v39]
v_pk_fma_f16 v154, v108, -1.0, v17 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v155, v107, -1.0, v106 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v156, v108, 1.0, v109 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v157, v107, 1.0, v110 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
s_setprio 2
label_79:
s_sub_u32 s3, s3, 4
s_cselect_b32 s18, 0, s18
ds_read_b64 [v92, v93], v2 offset:39296
buffer_load_short_d16 v17, v12, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v3, [v154, v155] offset:16512
buffer_load_short_d16 v108, v14, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v96, v97], v2 offset:39304
buffer_load_short_d16 v106, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16 v107, v15, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s48
s_addc_u32 s17, s17, 0
s_sub_u32 s3, s3, 4
s_cselect_b32 s18, 0, s18
ds_write_b64 v3, [v156, v157] offset:16768
buffer_load_short_d16 v129, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v104, v105], v2 offset:39312
buffer_load_short_d16 v132, v14, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16 v130, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16 v131, v15, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s48
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_35
s_mov_b32 s29, label_35 - label_1
s_branch label_46
.align 32
label_35:
s_setprio 1
s_waitcnt lgkmcnt(0)
s_barrier
v_pk_add_f16 v100, v92, v104 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v101, v93, v105 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mfma_f32_16x16x16_f16 [v26, v27, v28, v29], [v92, v93], [v34, v35], [v26, v27, v28, v29]
v_pk_mul_f16 v100, v100, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_pk_mul_f16 v101, v101, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_mfma_f32_16x16x16_f16 [v78, v79, v80, v81], [v104, v105], [v88, v89], [v78, v79, v80, v81]
v_pk_fma_f16 v100, v96, -0.5, v100 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v101, v97, -0.5, v101 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v154, v109, -2.0, v156 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v155, v110, -2.0, v157 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_mfma_f32_16x16x16_f16 [v60, v61, v62, v63], [v100, v101], [v86, v87], [v60, v61, v62, v63]
v_pk_fma_f16 v96, v96, 1.0, v100 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v97, v97, 1.0, v101 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v156, v109, -1.0, v112 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v157, v110, -1.0, v111 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_mfma_f32_16x16x16_f16 [v44, v45, v46, v47], [v96, v97], [v68, v69], [v44, v45, v46, v47]
s_setprio 2
label_80:
s_sub_u32 s3, s3, 4
s_cselect_b32 s18, 0, s18
ds_read_b64 [v34, v35], v1 offset:24768
buffer_load_short_d16 v109, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v68, v69], v1 offset:28928
buffer_load_short_d16 v112, v14, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v3, [v154, v155] offset:17024
buffer_load_short_d16 v110, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16 v111, v15, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s48
s_addc_u32 s17, s17, 0
s_sub_u32 s3, s3, 4
s_cselect_b32 s18, 0, s18
ds_read_b64 [v86, v87], v1 offset:24776
buffer_load_short_d16 v133, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v88, v89], v1 offset:28936
buffer_load_short_d16 v136, v14, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v3, [v156, v157] offset:17280
buffer_load_short_d16 v134, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16 v135, v15, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s48
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_36
s_mov_b32 s29, label_36 - label_1
s_branch label_47
.align 32
label_36:
s_mov_b32 s29, label_81 - label_1
s_sub_u32 s11, s11, 16
s_cbranch_scc1 label_51
label_81:
s_setprio 1
s_waitcnt vmcnt(32)
s_waitcnt lgkmcnt(0)
s_barrier
v_pk_fma_f16 v34, v86, -1.0, v34 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v35, v87, -1.0, v35 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v113, v113, v137 op_sel:[0, 0, 0]
v_pack_b32_f16 v114, v114, v138 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v22, v23, v24, v25], [v90, v91], [v34, v35], [v22, v23, v24, v25]
v_pk_fma_f16 v88, v68, -1.0, v88 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v89, v69, -1.0, v89 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v117, v117, v141 op_sel:[0, 0, 0]
v_pack_b32_f16 v118, v118, v142 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v74, v75, v76, v77], [v102, v103], [v88, v89], [v74, v75, v76, v77]
v_pk_fma_f16 v86, v68, -1.0, v86 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v87, v69, -1.0, v87 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v116, v116, v140 op_sel:[0, 0, 0]
v_pack_b32_f16 v115, v115, v139 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v56, v57, v58, v59], [v98, v99], [v86, v87], [v56, v57, v58, v59]
v_pk_fma_f16 v68, v68, 2.0, v86 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v69, v69, 2.0, v87 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v120, v120, v144 op_sel:[0, 0, 0]
v_pack_b32_f16 v119, v119, v143 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v40, v41, v42, v43], [v94, v95], [v68, v69], [v40, v41, v42, v43]
v_pk_fma_f16 v154, v116, -1.0, v113 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v155, v115, -1.0, v114 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v156, v116, 1.0, v117 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v157, v115, 1.0, v118 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
s_setprio 2
label_82:
s_sub_u32 s2, s2, 4
s_cselect_b32 s14, 0, s14
ds_read_b64 [v90, v91], v2 offset:33024
buffer_load_short_d16 v113, v8, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v3, [v154, v155] offset:24768
buffer_load_short_d16 v116, v10, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v94, v95], v2 offset:33032
buffer_load_short_d16 v114, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16 v115, v11, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s48
s_addc_u32 s13, s13, 0
s_sub_u32 s2, s2, 4
s_cselect_b32 s14, 0, s14
ds_write_b64 v3, [v156, v157] offset:25024
buffer_load_short_d16 v137, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v102, v103], v2 offset:33040
buffer_load_short_d16 v140, v10, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16 v138, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16 v139, v11, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s48
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_37
s_mov_b32 s29, label_37 - label_1
s_branch label_48
.align 32
label_37:
s_setprio 1
s_waitcnt lgkmcnt(0)
s_barrier
v_pk_add_f16 v98, v90, v102 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v99, v91, v103 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mfma_f32_16x16x16_f16 [v30, v31, v32, v33], [v92, v93], [v34, v35], [v30, v31, v32, v33]
v_pk_mul_f16 v98, v98, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_pk_mul_f16 v99, v99, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_mfma_f32_16x16x16_f16 [v82, v83, v84, v85], [v104, v105], [v88, v89], [v82, v83, v84, v85]
v_pk_fma_f16 v98, v94, -0.5, v98 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v99, v95, -0.5, v99 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v154, v117, -2.0, v156 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v155, v118, -2.0, v157 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_mfma_f32_16x16x16_f16 [v64, v65, v66, v67], [v100, v101], [v86, v87], [v64, v65, v66, v67]
v_pk_fma_f16 v94, v94, 1.0, v98 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v95, v95, 1.0, v99 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v156, v117, -1.0, v120 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v157, v118, -1.0, v119 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_mfma_f32_16x16x16_f16 [v48, v49, v50, v51], [v96, v97], [v68, v69], [v48, v49, v50, v51]
s_setprio 2
label_83:
s_sub_u32 s2, s2, 4
s_cselect_b32 s14, 0, s14
ds_read_b64 [v34, v35], v1 offset:16512
buffer_load_short_d16 v117, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v68, v69], v1 offset:20672
buffer_load_short_d16 v120, v10, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v3, [v154, v155] offset:25280
buffer_load_short_d16 v118, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16 v119, v11, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s48
s_addc_u32 s13, s13, 0
s_sub_u32 s2, s2, 4
s_cselect_b32 s14, 0, s14
ds_read_b64 [v86, v87], v1 offset:16520
buffer_load_short_d16 v141, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v88, v89], v1 offset:20680
buffer_load_short_d16 v144, v10, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v3, [v156, v157] offset:25536
buffer_load_short_d16 v142, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16 v143, v11, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s48
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_38
s_mov_b32 s29, label_38 - label_1
s_branch label_49
.align 32
label_38:
s_lshl_b32 s20, s20, 2
s_setprio 1
s_waitcnt vmcnt(32)
s_waitcnt lgkmcnt(0)
s_barrier
v_pk_fma_f16 v34, v86, -1.0, v34 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v35, v87, -1.0, v35 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v122, v122, v146 op_sel:[0, 0, 0]
v_pack_b32_f16 v124, v124, v148 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v18, v19, v20, v21], [v90, v91], [v34, v35], [v18, v19, v20, v21]
v_pk_fma_f16 v88, v68, -1.0, v88 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v89, v69, -1.0, v89 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v126, v126, v150 op_sel:[0, 0, 0]
v_pack_b32_f16 v128, v128, v152 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v70, v71, v72, v73], [v102, v103], [v88, v89], [v70, v71, v72, v73]
v_pk_fma_f16 v86, v68, -1.0, v86 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v87, v69, -1.0, v87 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v123, v123, v147 op_sel:[0, 0, 0]
v_pack_b32_f16 v121, v121, v145 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v52, v53, v54, v55], [v98, v99], [v86, v87], [v52, v53, v54, v55]
v_pk_fma_f16 v68, v68, 2.0, v86 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v69, v69, 2.0, v87 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v127, v127, v151 op_sel:[0, 0, 0]
v_pack_b32_f16 v125, v125, v149 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v36, v37, v38, v39], [v94, v95], [v68, v69], [v36, v37, v38, v39]
v_pk_fma_f16 v154, v123, -1.0, v122 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v155, v121, -1.0, v124 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v156, v123, 1.0, v126 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v157, v121, 1.0, v128 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
s_setprio 2
label_84:
s_sub_u32 s3, s3, 4
s_cselect_b32 s18, 0, s18
ds_read_b64 [v92, v93], v2 offset:39296
buffer_load_short_d16 v122, v12, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v3, [v154, v155] offset:16512
buffer_load_short_d16 v123, v14, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v96, v97], v2 offset:39304
buffer_load_short_d16 v124, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16 v121, v15, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s48
s_addc_u32 s17, s17, 0
s_sub_u32 s3, s3, 4
s_cselect_b32 s18, 0, s18
ds_write_b64 v3, [v156, v157] offset:16768
buffer_load_short_d16 v146, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v104, v105], v2 offset:39312
buffer_load_short_d16 v147, v14, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16 v148, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16 v145, v15, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s48
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_39
s_mov_b32 s29, label_39 - label_1
s_branch label_46
.align 32
label_39:
s_setprio 1
s_waitcnt lgkmcnt(0)
s_barrier
v_pk_add_f16 v100, v92, v104 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v101, v93, v105 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mfma_f32_16x16x16_f16 [v26, v27, v28, v29], [v92, v93], [v34, v35], [v26, v27, v28, v29]
v_pk_mul_f16 v100, v100, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_pk_mul_f16 v101, v101, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_mfma_f32_16x16x16_f16 [v78, v79, v80, v81], [v104, v105], [v88, v89], [v78, v79, v80, v81]
v_pk_fma_f16 v100, v96, -0.5, v100 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v101, v97, -0.5, v101 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v154, v126, -2.0, v156 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v155, v128, -2.0, v157 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_mfma_f32_16x16x16_f16 [v60, v61, v62, v63], [v100, v101], [v86, v87], [v60, v61, v62, v63]
v_pk_fma_f16 v96, v96, 1.0, v100 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v97, v97, 1.0, v101 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v156, v126, -1.0, v127 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v157, v128, -1.0, v125 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_mfma_f32_16x16x16_f16 [v44, v45, v46, v47], [v96, v97], [v68, v69], [v44, v45, v46, v47]
s_setprio 2
label_85:
s_sub_u32 s3, s3, 4
s_cselect_b32 s18, 0, s18
ds_read_b64 [v34, v35], v1 offset:24768
buffer_load_short_d16 v126, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v68, v69], v1 offset:28928
buffer_load_short_d16 v127, v14, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v3, [v154, v155] offset:17024
buffer_load_short_d16 v128, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16 v125, v15, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s48
s_addc_u32 s17, s17, 0
s_sub_u32 s3, s3, 4
s_cselect_b32 s18, 0, s18
ds_read_b64 [v86, v87], v1 offset:24776
buffer_load_short_d16 v150, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v88, v89], v1 offset:28936
buffer_load_short_d16 v151, v14, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v3, [v156, v157] offset:17280
buffer_load_short_d16 v152, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16 v149, v15, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s48
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_40
s_mov_b32 s29, label_40 - label_1
s_branch label_47
.align 32
label_40:
s_mov_b32 s29, label_86 - label_1
s_sub_u32 s11, s11, 16
s_cbranch_scc1 label_51
label_86:
s_setprio 1
s_waitcnt vmcnt(32)
s_waitcnt lgkmcnt(0)
s_barrier
v_pk_fma_f16 v34, v86, -1.0, v34 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v35, v87, -1.0, v35 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v17, v17, v129 op_sel:[0, 0, 0]
v_pack_b32_f16 v109, v109, v133 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v22, v23, v24, v25], [v90, v91], [v34, v35], [v22, v23, v24, v25]
v_pk_fma_f16 v88, v68, -1.0, v88 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v89, v69, -1.0, v89 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v106, v106, v130 op_sel:[0, 0, 0]
v_pack_b32_f16 v110, v110, v134 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v74, v75, v76, v77], [v102, v103], [v88, v89], [v74, v75, v76, v77]
v_pk_fma_f16 v86, v68, -1.0, v86 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v87, v69, -1.0, v87 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v108, v108, v132 op_sel:[0, 0, 0]
v_pack_b32_f16 v112, v112, v136 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v56, v57, v58, v59], [v98, v99], [v86, v87], [v56, v57, v58, v59]
v_pk_fma_f16 v68, v68, 2.0, v86 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v69, v69, 2.0, v87 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v107, v107, v131 op_sel:[0, 0, 0]
v_pack_b32_f16 v111, v111, v135 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v40, v41, v42, v43], [v94, v95], [v68, v69], [v40, v41, v42, v43]
v_pk_fma_f16 v154, v108, -1.0, v17 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v155, v112, -1.0, v109 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v156, v108, 1.0, v106 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v157, v112, 1.0, v110 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
s_setprio 2
label_87:
s_sub_u32 s2, s2, 4
s_cselect_b32 s14, 0, s14
ds_read_b64 [v90, v91], v2 offset:33024
buffer_load_short_d16 v17, v8, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v3, [v154, v155] offset:24768
buffer_load_short_d16 v108, v10, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v94, v95], v2 offset:33032
buffer_load_short_d16 v109, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16 v112, v11, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s48
s_addc_u32 s13, s13, 0
s_sub_u32 s2, s2, 4
s_cselect_b32 s14, 0, s14
ds_write_b64 v3, [v156, v157] offset:25024
buffer_load_short_d16 v129, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v102, v103], v2 offset:33040
buffer_load_short_d16 v132, v10, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16 v133, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16 v136, v11, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s48
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_41
s_mov_b32 s29, label_41 - label_1
s_branch label_48
.align 32
label_41:
s_setprio 1
s_waitcnt lgkmcnt(0)
s_barrier
v_pk_add_f16 v98, v90, v102 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v99, v91, v103 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mfma_f32_16x16x16_f16 [v30, v31, v32, v33], [v92, v93], [v34, v35], [v30, v31, v32, v33]
v_pk_mul_f16 v98, v98, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_pk_mul_f16 v99, v99, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_mfma_f32_16x16x16_f16 [v82, v83, v84, v85], [v104, v105], [v88, v89], [v82, v83, v84, v85]
v_pk_fma_f16 v98, v94, -0.5, v98 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v99, v95, -0.5, v99 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v154, v106, -2.0, v156 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v155, v110, -2.0, v157 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_mfma_f32_16x16x16_f16 [v64, v65, v66, v67], [v100, v101], [v86, v87], [v64, v65, v66, v67]
v_pk_fma_f16 v94, v94, 1.0, v98 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v95, v95, 1.0, v99 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v156, v106, -1.0, v107 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v157, v110, -1.0, v111 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_mfma_f32_16x16x16_f16 [v48, v49, v50, v51], [v96, v97], [v68, v69], [v48, v49, v50, v51]
s_setprio 2
label_88:
s_sub_u32 s2, s2, 4
s_cselect_b32 s14, 0, s14
ds_read_b64 [v34, v35], v1 offset:16512
buffer_load_short_d16 v106, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v68, v69], v1 offset:20672
buffer_load_short_d16 v107, v10, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v3, [v154, v155] offset:25280
buffer_load_short_d16 v110, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16 v111, v11, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s48
s_addc_u32 s13, s13, 0
s_sub_u32 s2, s2, 4
s_cselect_b32 s14, 0, s14
ds_read_b64 [v86, v87], v1 offset:16520
buffer_load_short_d16 v130, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v88, v89], v1 offset:20680
buffer_load_short_d16 v131, v10, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v3, [v156, v157] offset:25536
buffer_load_short_d16 v134, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16 v135, v11, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s48
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_42
s_mov_b32 s29, label_42 - label_1
s_branch label_49
.align 32
label_42:
s_lshl_b32 s20, s20, 2
s_setprio 1
s_waitcnt vmcnt(32)
s_waitcnt lgkmcnt(0)
s_barrier
v_pk_fma_f16 v34, v86, -1.0, v34 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v35, v87, -1.0, v35 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v113, v113, v137 op_sel:[0, 0, 0]
v_pack_b32_f16 v117, v117, v141 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v18, v19, v20, v21], [v90, v91], [v34, v35], [v18, v19, v20, v21]
v_pk_fma_f16 v88, v68, -1.0, v88 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v89, v69, -1.0, v89 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v114, v114, v138 op_sel:[0, 0, 0]
v_pack_b32_f16 v118, v118, v142 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v70, v71, v72, v73], [v102, v103], [v88, v89], [v70, v71, v72, v73]
v_pk_fma_f16 v86, v68, -1.0, v86 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v87, v69, -1.0, v87 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v116, v116, v140 op_sel:[0, 0, 0]
v_pack_b32_f16 v120, v120, v144 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v52, v53, v54, v55], [v98, v99], [v86, v87], [v52, v53, v54, v55]
v_pk_fma_f16 v68, v68, 2.0, v86 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v69, v69, 2.0, v87 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v115, v115, v139 op_sel:[0, 0, 0]
v_pack_b32_f16 v119, v119, v143 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v36, v37, v38, v39], [v94, v95], [v68, v69], [v36, v37, v38, v39]
v_pk_fma_f16 v154, v116, -1.0, v113 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v155, v120, -1.0, v117 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v156, v116, 1.0, v114 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v157, v120, 1.0, v118 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
s_setprio 2
label_89:
s_sub_u32 s3, s3, 4
s_cselect_b32 s18, 0, s18
ds_read_b64 [v92, v93], v2 offset:39296
buffer_load_short_d16 v113, v12, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v3, [v154, v155] offset:16512
buffer_load_short_d16 v116, v14, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v96, v97], v2 offset:39304
buffer_load_short_d16 v117, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16 v120, v15, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s48
s_addc_u32 s17, s17, 0
s_sub_u32 s3, s3, 4
s_cselect_b32 s18, 0, s18
ds_write_b64 v3, [v156, v157] offset:16768
buffer_load_short_d16 v137, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v104, v105], v2 offset:39312
buffer_load_short_d16 v140, v14, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16 v141, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16 v144, v15, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s48
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_43
s_mov_b32 s29, label_43 - label_1
s_branch label_46
.align 32
label_43:
s_setprio 1
s_waitcnt lgkmcnt(0)
s_barrier
v_pk_add_f16 v100, v92, v104 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v101, v93, v105 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mfma_f32_16x16x16_f16 [v26, v27, v28, v29], [v92, v93], [v34, v35], [v26, v27, v28, v29]
v_pk_mul_f16 v100, v100, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_pk_mul_f16 v101, v101, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_mfma_f32_16x16x16_f16 [v78, v79, v80, v81], [v104, v105], [v88, v89], [v78, v79, v80, v81]
v_pk_fma_f16 v100, v96, -0.5, v100 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v101, v97, -0.5, v101 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v154, v114, -2.0, v156 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v155, v118, -2.0, v157 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_mfma_f32_16x16x16_f16 [v60, v61, v62, v63], [v100, v101], [v86, v87], [v60, v61, v62, v63]
v_pk_fma_f16 v96, v96, 1.0, v100 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v97, v97, 1.0, v101 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v156, v114, -1.0, v115 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v157, v118, -1.0, v119 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_mfma_f32_16x16x16_f16 [v44, v45, v46, v47], [v96, v97], [v68, v69], [v44, v45, v46, v47]
s_setprio 2
label_90:
s_sub_u32 s3, s3, 4
s_cselect_b32 s18, 0, s18
ds_read_b64 [v34, v35], v1 offset:24768
buffer_load_short_d16 v114, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v68, v69], v1 offset:28928
buffer_load_short_d16 v115, v14, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v3, [v154, v155] offset:17024
buffer_load_short_d16 v118, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16 v119, v15, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s48
s_addc_u32 s17, s17, 0
s_sub_u32 s3, s3, 4
s_cselect_b32 s18, 0, s18
ds_read_b64 [v86, v87], v1 offset:24776
buffer_load_short_d16 v138, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v88, v89], v1 offset:28936
buffer_load_short_d16 v139, v14, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v3, [v156, v157] offset:17280
buffer_load_short_d16 v142, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16 v143, v15, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s48
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_44
s_mov_b32 s29, label_44 - label_1
s_branch label_47
.align 32
label_44:
s_mov_b32 s29, label_91 - label_1
s_sub_u32 s11, s11, 16
s_cbranch_scc1 label_51
label_91:
s_setprio 1
s_waitcnt vmcnt(32)
s_waitcnt lgkmcnt(0)
s_barrier
v_pk_fma_f16 v34, v86, -1.0, v34 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v35, v87, -1.0, v35 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v122, v122, v146 op_sel:[0, 0, 0]
v_pack_b32_f16 v126, v126, v150 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v22, v23, v24, v25], [v90, v91], [v34, v35], [v22, v23, v24, v25]
v_pk_fma_f16 v88, v68, -1.0, v88 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v89, v69, -1.0, v89 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v124, v124, v148 op_sel:[0, 0, 0]
v_pack_b32_f16 v128, v128, v152 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v74, v75, v76, v77], [v102, v103], [v88, v89], [v74, v75, v76, v77]
v_pk_fma_f16 v86, v68, -1.0, v86 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v87, v69, -1.0, v87 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v123, v123, v147 op_sel:[0, 0, 0]
v_pack_b32_f16 v127, v127, v151 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v56, v57, v58, v59], [v98, v99], [v86, v87], [v56, v57, v58, v59]
v_pk_fma_f16 v68, v68, 2.0, v86 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v69, v69, 2.0, v87 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pack_b32_f16 v121, v121, v145 op_sel:[0, 0, 0]
v_pack_b32_f16 v125, v125, v149 op_sel:[0, 0, 0]
v_mfma_f32_16x16x16_f16 [v40, v41, v42, v43], [v94, v95], [v68, v69], [v40, v41, v42, v43]
v_pk_fma_f16 v154, v123, -1.0, v122 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v155, v127, -1.0, v126 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v156, v123, 1.0, v124 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v157, v127, 1.0, v128 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
s_setprio 2
label_92:
s_sub_u32 s2, s2, 4
s_cselect_b32 s14, 0, s14
ds_read_b64 [v90, v91], v2 offset:33024
buffer_load_short_d16 v122, v8, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v3, [v154, v155] offset:24768
buffer_load_short_d16 v123, v10, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v94, v95], v2 offset:33032
buffer_load_short_d16 v126, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16 v127, v11, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s48
s_addc_u32 s13, s13, 0
s_sub_u32 s2, s2, 4
s_cselect_b32 s14, 0, s14
ds_write_b64 v3, [v156, v157] offset:25024
buffer_load_short_d16 v146, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v102, v103], v2 offset:33040
buffer_load_short_d16 v147, v10, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16 v150, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16 v151, v11, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s48
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_45
s_mov_b32 s29, label_45 - label_1
s_branch label_48
.align 32
label_45:
s_setprio 1
s_waitcnt lgkmcnt(0)
s_barrier
v_pk_add_f16 v98, v90, v102 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v99, v91, v103 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mfma_f32_16x16x16_f16 [v30, v31, v32, v33], [v92, v93], [v34, v35], [v30, v31, v32, v33]
v_pk_mul_f16 v98, v98, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_pk_mul_f16 v99, v99, 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 0, 1]
v_mfma_f32_16x16x16_f16 [v82, v83, v84, v85], [v104, v105], [v88, v89], [v82, v83, v84, v85]
v_pk_fma_f16 v98, v94, -0.5, v98 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v99, v95, -0.5, v99 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v154, v124, -2.0, v156 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v155, v128, -2.0, v157 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_mfma_f32_16x16x16_f16 [v64, v65, v66, v67], [v100, v101], [v86, v87], [v64, v65, v66, v67]
v_pk_fma_f16 v94, v94, 1.0, v98 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v95, v95, 1.0, v99 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v156, v124, -1.0, v121 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_pk_fma_f16 v157, v128, -1.0, v125 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 1]
v_mfma_f32_16x16x16_f16 [v48, v49, v50, v51], [v96, v97], [v68, v69], [v48, v49, v50, v51]
s_setprio 2
label_93:
s_sub_u32 s2, s2, 4
s_cselect_b32 s14, 0, s14
ds_read_b64 [v34, v35], v1 offset:16512
buffer_load_short_d16 v124, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v68, v69], v1 offset:20672
buffer_load_short_d16 v121, v10, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v3, [v154, v155] offset:25280
buffer_load_short_d16 v128, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16 v125, v11, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s48
s_addc_u32 s13, s13, 0
s_sub_u32 s2, s2, 4
s_cselect_b32 s14, 0, s14
ds_read_b64 [v86, v87], v1 offset:16520
buffer_load_short_d16 v148, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v88, v89], v1 offset:20680
buffer_load_short_d16 v145, v10, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v3, [v156, v157] offset:25536
buffer_load_short_d16 v152, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16 v149, v11, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s48
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s20, 5
s_cbranch_scc0 label_34
s_mov_b32 s29, label_34 - label_1
s_branch label_49
.align 32
label_46:
v_add_f32 v18, v18, v36
v_add_f32 v19, v19, v37
v_add_f32 v20, v20, v38
v_add_f32 v21, v21, v39
v_add_f32 v70, v70, neg(v52)
v_add_f32 v71, v71, neg(v53)
v_add_f32 v72, v72, neg(v54)
v_add_f32 v73, v73, neg(v55)
v_add_f32 v18, v18, v52
v_add_f32 v19, v19, v53
v_add_f32 v20, v20, v54
v_add_f32 v21, v21, v55
v_add_f32 v70, v70, v36
v_add_f32 v71, v71, v37
v_add_f32 v72, v72, v38
v_add_f32 v73, v73, v39
v_cvt_f16_f32 v160, v18 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE
v_cvt_f16_f32 v160, v19 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE
v_cvt_f16_f32 v161, v20 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE
v_cvt_f16_f32 v161, v21 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE
v_cvt_f16_f32 v158, v70 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE
v_cvt_f16_f32 v158, v71 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE
v_cvt_f16_f32 v159, v72 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE
v_cvt_f16_f32 v159, v73 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE
v_mov_b32 v18, 0
v_mov_b32 v19, 0
v_mov_b32 v20, 0
v_mov_b32 v21, 0
v_mov_b32 v36, 0
v_mov_b32 v37, 0
v_mov_b32 v38, 0
v_mov_b32 v39, 0
v_mov_b32 v52, 0
v_mov_b32 v53, 0
v_mov_b32 v54, 0
v_mov_b32 v55, 0
v_mov_b32 v70, 0
v_mov_b32 v71, 0
v_mov_b32 v72, 0
v_mov_b32 v73, 0
ds_write_b64 v1, [v160, v161] offset:4160
ds_write_b64 v1, [v158, v159] offset:8
s_add_u32 s36, s6, s29
s_addc_u32 s37, s7, 0
s_setpc_b64 [s36, s37]
.align 32
label_47:
v_add_f32 v26, v26, v44
v_add_f32 v27, v27, v45
v_add_f32 v28, v28, v46
v_add_f32 v29, v29, v47
v_add_f32 v78, v78, neg(v60)
v_add_f32 v79, v79, neg(v61)
v_add_f32 v80, v80, neg(v62)
v_add_f32 v81, v81, neg(v63)
v_add_f32 v26, v26, v60
v_add_f32 v27, v27, v61
v_add_f32 v28, v28, v62
v_add_f32 v29, v29, v63
v_add_f32 v78, v78, v44
v_add_f32 v79, v79, v45
v_add_f32 v80, v80, v46
v_add_f32 v81, v81, v47
v_cvt_f16_f32 v160, v26 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE
v_cvt_f16_f32 v160, v27 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE
v_cvt_f16_f32 v161, v28 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE
v_cvt_f16_f32 v161, v29 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE
v_cvt_f16_f32 v158, v78 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE
v_cvt_f16_f32 v158, v79 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE
v_cvt_f16_f32 v159, v80 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE
v_cvt_f16_f32 v159, v81 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE
v_mov_b32 v26, 0
v_mov_b32 v27, 0
v_mov_b32 v28, 0
v_mov_b32 v29, 0
v_mov_b32 v44, 0
v_mov_b32 v45, 0
v_mov_b32 v46, 0
v_mov_b32 v47, 0
v_mov_b32 v60, 0
v_mov_b32 v61, 0
v_mov_b32 v62, 0
v_mov_b32 v63, 0
v_mov_b32 v78, 0
v_mov_b32 v79, 0
v_mov_b32 v80, 0
v_mov_b32 v81, 0
ds_write_b64 v1, [v160, v161] offset:12416
ds_write_b64 v1, [v158, v159] offset:8264
s_add_u32 s36, s6, s29
s_addc_u32 s37, s7, 0
s_setpc_b64 [s36, s37]
.align 32
label_48:
v_add_f32 v22, v22, v40
v_add_f32 v23, v23, v41
v_add_f32 v24, v24, v42
v_add_f32 v25, v25, v43
v_add_f32 v74, v74, neg(v56)
v_add_f32 v75, v75, neg(v57)
v_add_f32 v76, v76, neg(v58)
v_add_f32 v77, v77, neg(v59)
v_add_f32 v22, v22, v56
v_add_f32 v23, v23, v57
v_add_f32 v24, v24, v58
v_add_f32 v25, v25, v59
v_add_f32 v74, v74, v40
v_add_f32 v75, v75, v41
v_add_f32 v76, v76, v42
v_add_f32 v77, v77, v43
v_cvt_f16_f32 v160, v22 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE
v_cvt_f16_f32 v160, v23 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE
v_cvt_f16_f32 v161, v24 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE
v_cvt_f16_f32 v161, v25 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE
v_cvt_f16_f32 v158, v74 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE
v_cvt_f16_f32 v158, v75 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE
v_cvt_f16_f32 v159, v76 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE
v_cvt_f16_f32 v159, v77 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE
v_mov_b32 v22, 0
v_mov_b32 v23, 0
v_mov_b32 v24, 0
v_mov_b32 v25, 0
v_mov_b32 v40, 0
v_mov_b32 v41, 0
v_mov_b32 v42, 0
v_mov_b32 v43, 0
v_mov_b32 v56, 0
v_mov_b32 v57, 0
v_mov_b32 v58, 0
v_mov_b32 v59, 0
v_mov_b32 v74, 0
v_mov_b32 v75, 0
v_mov_b32 v76, 0
v_mov_b32 v77, 0
ds_write_b64 v1, [v160, v161] offset:4160
ds_write_b64 v1, [v158, v159] offset:8
s_add_u32 s36, s6, s29
s_addc_u32 s37, s7, 0
s_setpc_b64 [s36, s37]
.align 32
label_49:
v_add_f32 v30, v30, v48
v_add_f32 v31, v31, v49
v_add_f32 v32, v32, v50
v_add_f32 v33, v33, v51
v_add_f32 v82, v82, neg(v64)
v_add_f32 v83, v83, neg(v65)
v_add_f32 v84, v84, neg(v66)
v_add_f32 v85, v85, neg(v67)
v_add_f32 v30, v30, v64
v_add_f32 v31, v31, v65
v_add_f32 v32, v32, v66
v_add_f32 v33, v33, v67
v_add_f32 v82, v82, v48
v_add_f32 v83, v83, v49
v_add_f32 v84, v84, v50
v_add_f32 v85, v85, v51
v_cvt_f16_f32 v160, v30 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE
v_cvt_f16_f32 v160, v31 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE
v_cvt_f16_f32 v161, v32 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE
v_cvt_f16_f32 v161, v33 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE
v_cvt_f16_f32 v158, v82 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE
v_cvt_f16_f32 v158, v83 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE
v_cvt_f16_f32 v159, v84 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE
v_cvt_f16_f32 v159, v85 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE
v_mov_b32 v30, 0
v_mov_b32 v31, 0
v_mov_b32 v32, 0
v_mov_b32 v33, 0
v_mov_b32 v48, 0
v_mov_b32 v49, 0
v_mov_b32 v50, 0
v_mov_b32 v51, 0
v_mov_b32 v64, 0
v_mov_b32 v65, 0
v_mov_b32 v66, 0
v_mov_b32 v67, 0
v_mov_b32 v82, 0
v_mov_b32 v83, 0
v_mov_b32 v84, 0
v_mov_b32 v85, 0
ds_write_b64 v1, [v160, v161] offset:12416
ds_write_b64 v1, [v158, v159] offset:8264
s_bitcmp1_b32 s20, 4
s_cmov_b32 s29, label_0 - label_1
s_add_u32 s36, s6, s29
s_addc_u32 s37, s7, 0
s_setpc_b64 [s36, s37]
label_5:
s_and_b32 s45, s45, 65535
s_add_u32 s45, s45, 131072
s_mov_b64 [s80, s81], [s44, s45]
s_mov_b32 s82, 2147483648
s_mov_b32 s83, 131072
s_bitcmp0_b32 s28, 7
s_cselect_b32 s82, 0, 2147483648
s_and_b32 s79, s1, 3
v_and_b32 v87, v0, 63
v_lshrrev_b32 v86, 1, v87
v_and_b32 v85, v87, 1
v_mad_u32_u24 v85, v86, 8, v85
v_lshlrev_b32 v85, 1, v85
v_mad_u32_u24 v85, 4, s79, v85
v_add_u32 v86, v85, 1
v_cmp_lt_u32 [vcc_lo, vcc_hi], v85, s26
v_cndmask_b32 v85, -1, v85, [vcc_lo, vcc_hi]
v_cmp_lt_u32 [vcc_lo, vcc_hi], v86, s26
v_cndmask_b32 v86, -1, v86, [vcc_lo, vcc_hi]
buffer_load_short_d16 v77, v85, [s80, s81, s82, s83], 0 idxen
buffer_load_short_d16 v81, v86, [s80, s81, s82, s83], 0 idxen
s_cmp_eq_u32 s1, 9
s_cselect_b32 s44, 1536, 0
s_cselect_b32 s45, 12, 0
s_cmp_ge_u32 s1, 10
s_cselect_b32 s44, 65536, s44
s_cselect_b32 s45, 65536, s45
s_cmp_eq_u32 s1, 9
s_cselect_b32 s79, 1024, 0
s_cselect_b32 s80, 4, 0
s_cmp_ge_u32 s1, 10
s_cselect_b32 s79, 65536, s79
s_cselect_b32 s80, 65536, s80
s_and_b32 s82, s1, 3
s_mul_i32 s81, s82, 2048
v_and_b32 v85, 63, v0
v_lshl_add_u32 v54, v85, 3, s44
v_lshl_add_u32 v71, v85, 3, 0
v_mov_b32 v72, s45
v_mov_b32 v73, 0
v_lshl_add_u32 v74, v85, 4, s79
v_mov_b32 v76, s80
v_lshl_add_u32 v75, v85, 4, s81
s_add_u32 s80, s23, 15
s_and_b32 s80, s80, -16
s_sub_u32 s80, s80, 16
s_sub_u32 s80, s23, s80
s_bitcmp1_b32 s28, 1
s_cselect_b32 s81, s39, 1
s_cselect_b32 s29, -1, 1
s_sub_u32 s79, s81, 1
s_bitcmp1_b32 s28, 0
s_cselect_b32 s81, s38, 1
s_cselect_b32 s19, -1, 1
s_sub_u32 s81, s81, 1
s_mul_i32 s19, s19, s62
s_mul_i32 s81, s81, s62
s_add_u32 s79, s79, s81
s_bitcmp1_b32 s28, 31
s_cselect_b32 s45, 0, 1
s_cselect_b32 s44, 4, 0
s_cselect_b32 s81, 2, 0
s_cselect_b32 s82, 2, 4
v_bfe_u32 v86, v0, 2, 6
v_and_b32 v87, s1, 3
v_bfe_u32 v85, v86, 0, s81
v_mad_u32_u24 v85, v87, s45, v85
v_mov_b32 v10, s82
v_bfe_u32 v10, v86, s81, v10
v_mad_u32_u24 v10, v87, s44, v10
v_mov_b32 v86, s79
v_mad_u32_u24 v5, s61, v85, v86
v_sub_u32 v21, s80, v85
v_add_u32 v17, v21, 3
v_add_u32 v21, v21, 3
s_add_u32 s45, s42, 1
s_lshr_b32 s45, s45, 1
s_lshl_b32 s61, s45, 1
s_add_u32 s62, s43, 1
s_lshr_b32 s62, s62, 1
s_lshl1_add_u32 s62, s62, 2
s_pack_ll_b32_b16 s3, s62, s45
s_pack_ll_b32_b16 s10, s25, s24
s_sub_u32 s11, s61, s40
s_sub_u32 s44, s62, s41
s_pack_ll_b32_b16 s11, s44, s11
s_pack_ll_b32_b16 s13, s43, s42
s_sub_u32 s44, s62, 1
s_pack_ll_b32_b16 s14, s44, s61
v_lshrrev_b32 v88, 16, s3
v_mov_b32 v103, 65535
v_bfi_b32 v89, v103, s3, 0
v_and_b32 v91, 1, v0
v_bfe_u32 v97, v0, 6, 1
v_and_b32 v86, 63, v0
v_mov_b32 v103, 124
v_mad_u32_u16 v92, v103, s5, 0 op_sel:[0, 0, 0, 0]
v_mov_b32 v103, 2
v_mad_u32_u16 v97, v103, s8, v97 op_sel:[0, 0, 0, 0]
v_mad_u32_u16 v90, v88, v89, 0 op_sel:[0, 0, 0, 0]
v_cmp_eq_u32 [vcc_lo, vcc_hi], 0, v91
v_cndmask_b32 v98, v90, v89, [vcc_lo, vcc_hi]
v_mad_u32_u16 v87, 62, v97, v86 op_sel:[0, 0, 0, 0]
v_cndmask_b32 v87, v92, v87, [vcc_lo, vcc_hi]
v_ffbh_u32 v104, v98
v_lshlrev_b32 v105, v104, v98
v_and_b32 v103, -256, v105
v_cmp_eq_u32 [vcc_lo, vcc_hi], 2147483648, v105
v_cvt_f32_u32 v103, v103
v_rcp_f32 v99, v103
v_subb_co_u32 v100, [vcc_lo, vcc_hi], 32, v104, [vcc_lo, vcc_hi]
v_cvt_f32_ubyte0 v104, v105
v_fma_f32 v103, v103, v99, -1.0
v_fma_f32 v103, v104, v99, v103
v_fmaak_f32 v103, v103, v99, 2667577344
v_mul_f32 v103, 1602224128, v103
v_mov_b32 v104, 0
v_cvt_flr_i32_f32 v103, neg(v103)
v_lshl_add_u32 v99, v99, 9, v103
v_mad_u64_u32 [v104, v105], [vcc_lo, vcc_hi], v105, v99, [v104, v105]
v_subb_co_u32 v99, [vcc_lo, vcc_hi], v99, -1, [vcc_lo, vcc_hi]
v_mov_b32 v102, v100 quad_perm:[1, 1, 1, 1]
v_mov_b32 v100, v100 quad_perm:[0, 0, 0, 0]
v_mov_b32 v101, v99 quad_perm:[1, 1, 1, 1]
v_mov_b32 v99, v99 quad_perm:[0, 0, 0, 0]
v_mul_hi_u32 v103, v87, v99
v_add_co_u32 v85, [vcc_lo, vcc_hi], v103, v87
v_addc_co_u32 v103, [vcc_lo, vcc_hi], 0, 0, [vcc_lo, vcc_hi]
v_cmp_eq_u32 [vcc_lo, vcc_hi], 32, v100
v_cndmask_b32 v85, v85, v103, [vcc_lo, vcc_hi]
v_alignbit_b32 v85, v103, v85, v100
v_mul_hi_u32 v103, v87, v101
v_add_co_u32 v4, [vcc_lo, vcc_hi], v103, v87
v_addc_co_u32 v103, [vcc_lo, vcc_hi], 0, 0, [vcc_lo, vcc_hi]
v_cmp_eq_u32 [vcc_lo, vcc_hi], 32, v102
v_cndmask_b32 v4, v4, v103, [vcc_lo, vcc_hi]
v_alignbit_b32 v4, v103, v4, v102
v_mad_u32_u16 v96, v85, v89, 0 op_sel:[0, 0, 0, 0]
v_mad_u32_u16 v95, v4, v88, 0 op_sel:[0, 0, 0, 0]
v_sub_u32 v96, v87, v96
v_sub_u32 v95, v85, v95
v_readlane_b32 s79, v96, 1
v_sub_u32 v96, v96, v89
v_readlane_b32 s9, v95, 1
v_sub_u32 v95, v95, v88
v_readlane_b32 s2, v4, 1
v_sub_u32 v4, v4, s22
s_lshl_b32 s9, s9, 16
s_and_b32 s79, s79, lit(0xffff)
s_add_u32 s9, s9, s79
v_mov_b32 v96, v96 quad_perm:[0, 0, 2, 2]
v_mov_b32 v95, v95 quad_perm:[0, 0, 2, 2]
v_mov_b32 v4, v4 quad_perm:[0, 0, 2, 2]
v_add_co_u32 v96, [vcc_lo, vcc_hi], v96, v91
v_cndmask_b32 v94, 0, v89, [vcc_lo, vcc_hi]
v_addc_co_u32 v95, [vcc_lo, vcc_hi], v95, 0, [vcc_lo, vcc_hi]
v_cndmask_b32 v93, 0, v88, [vcc_lo, vcc_hi]
v_addc_co_u32 v4, [vcc_lo, vcc_hi], v4, 0, [vcc_lo, vcc_hi]
v_min_u32 v91, v86, 63
v_sub_u32 v96, v96, v94
v_sub_u32 v95, v95, v93
v_cmp_eq_u32 [vcc_lo, vcc_hi], v86, v91
v_lshlrev_b32 v1, 16, v95
v_mov_b32 v106, 65535
v_bfi_b32 v1, v106, v96, v1
v_add_u32 v106, v4, s22
v_med3_u32 v91, v86, 1, 62
v_mul_lo_u32 v2, v106, s56
v_mul_lo_u32 v11, v106, s64
s_mul_i32 s12, s2, s56
s_mul_i32 s15, s2, s64
v_mov_b32 v107, 2147483648
v_cndmask_b32 v2, v107, v2, [vcc_lo, vcc_hi]
v_cmp_eq_u32 [vcc_lo, vcc_hi], v86, v91
v_cndmask_b32 v11, v107, v11, [vcc_lo, vcc_hi]
v_cmp_ge_u32 [s20, s21], v106, s22
v_cndmask_b32 v2, v2, v107, [s20, s21]
v_cndmask_b32 v11, v11, v107, [s20, s21]
s_and_b32 s24, s72, 255
s_cmp_eq_u32 s24, 0
s_cmov_b32 s46, lit(0x3c003c00)
s_cbranch_scc1 label_118
s_cmp_eq_u32 s24, 4
s_cbranch_scc1 label_120
s_cmp_eq_u32 s24, 2
s_cbranch_scc1 label_121
v_cvt_f16_f32 v85, s46
s_nop 0
v_readfirstlane_b32 s46, v85
v_cvt_f16_f32 v85, s47
v_cmp_gt_f16 [vcc_lo, vcc_hi], s46, 1.0
v_readfirstlane_b32 s47, v85
s_pack_ll_b32_b16 s46, s46, s46
s_pack_ll_b32_b16 s47, s47, s47
s_cmp_eq_u32 s24, 3
s_cbranch_scc1 label_122
s_cbranch_vccnz label_119
label_118:
s_mov_b32 s78, label_98 - label_1
s_branch label_123
label_119:
s_mov_b32 s78, label_102 - label_1
s_branch label_123
label_120:
s_mov_b32 s78, label_106 - label_1
s_branch label_123
label_121:
s_mov_b32 s78, label_110 - label_1
s_branch label_123
label_122:
s_mov_b32 s78, label_114 - label_1
label_123:
s_lshl_b32 s59, s65, 1
s_mul_i32 s54, s59, 16
s_mul_hi_u32 s55, s59, 16
s_and_b32 s24, s1, 3
s_lshl_b32 s25, s59, 2
s_mul_i32 s25, s25, s24
s_add_u32 s34, s34, s25
s_addc_u32 s35, s35, 0
s_lshl_b32 s67, s24, 2
s_sub_u32 s67, s26, s67
s_cmov_b32 s67, 0
s_and_b32 s35, s35, 65535
s_add_u32 s35, s35, 131072
s_mov_b64 [s68, s69], [s34, s35]
s_mov_b32 s70, 0
s_mov_b32 s71, 131072
s_mov_b32 s16, 65536
s_mov_b32 s17, 196610
s_mov_b32 s18, 65536
s_add_u32 s30, s38, 2
s_add_u32 s31, s39, 2
s_mul_hi_u32 s25, s30, 1431655766
s_add_u32 s30, s25, s30
s_addc_u32 s25, 0, 0
s_mov_b32 s24, s30
s_lshr_b64 [s24, s25], [s24, s25], 2
s_mov_b32 s30, s24
s_mul_hi_u32 s25, s31, 1431655766
s_add_u32 s31, s25, s31
s_addc_u32 s25, 0, 0
s_mov_b32 s24, s31
s_lshr_b64 [s24, s25], [s24, s25], 2
s_mov_b32 s31, s24
s_sub_u32 s30, 0, s30
s_sub_u32 s31, 0, s31
s_mov_b32 s32, -1
s_mov_b32 s33, -1
s_waitcnt vmcnt(0)
v_pack_b32_f16 v77, v77, v81 op_sel:[0, 0, 0]
s_nop 0
v_readlane_b32 s36, v77, 0
v_readlane_b32 s37, v77, 1
v_readlane_b32 s48, v77, 2
v_readlane_b32 s49, v77, 3
s_mov_b32 s73, 0
s_mov_b32 s74, 0
s_mov_b32 s50, 2
s_lshl_b32 s51, s50, 9
s_lshl_b32 s52, s50, 2
s_branch label_97
.align 32
label_94:
s_sub_u32 s74, s23, 1
s_add_u32 s76, s6, s75
s_addc_u32 s77, s7, 0
s_setprio 3
s_add_u32 s33, s33, 1
s_cselect_b32 s33, s31, s33
s_addc_u32 s32, s32, 0
s_cselect_b32 s32, s30, s32
s_cbranch_scc0 label_124
s_bitcmp1_b32 s20, 0
s_addc_u32 s73, s73, 2
v_add_u32 v77, s51, v54
v_add_u32 v81, s52, v72
v_and_b32 v84, s73, 3
ds_write_b64 v77, [v12, v13] offset:55816
ds_write_b32 v81, v84 offset:58888
label_124:
s_sub_u32 s24, -1, s32
s_sub_u32 s25, -1, s33
s_mul_i32 s24, 3, s24
s_mul_i32 s25, 3, s25
s_pack_ll_b32_b16 s40, s25, s24
s_cmp_eq_u32 s40, 0
s_cbranch_scc1 label_126
label_125:
v_mov_b32 v81, 2147483648
v_mov_b32 v87, 131073
v_pk_mad_u16 v87, v1, v87, s11 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_add_u16 v87, s40, v87 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mad_u32_u16 v6, v87, 1, v2 op_sel:[0, 0, 0, 0]
v_cmp_lt_u16 [vcc_lo, vcc_hi], v87, s10
v_cndmask_b32 v6, v81, v6, [vcc_lo, vcc_hi]
v_pk_add_u16 v77, v87, s17 op_sel:[1, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_u16 v88, v87, s16 op_sel:[1, 0, 0] op_sel_hi:[1, 1, 1]
v_mad_u32_u16 v9, v77, s58, v6 op_sel:[1, 0, 0, 0]
v_mad_u32_u16 v7, v88, s58, v6 op_sel:[1, 0, 0, 0]
v_cmp_lt_u16 [vcc_lo, vcc_hi], v77, s10 src0_sel:WORD_1 src1_sel:WORD_1
v_cndmask_b32 v9, v81, v9, [vcc_lo, vcc_hi]
v_cmp_lt_u16 [vcc_lo, vcc_hi], v88, s10 src0_sel:WORD_1 src1_sel:WORD_1
v_cndmask_b32 v7, v81, v7, [vcc_lo, vcc_hi]
v_mad_u32_u16 v8, v77, s58, v6 op_sel:[0, 0, 0, 0]
v_mad_u32_u16 v6, v88, s58, v6 op_sel:[0, 0, 0, 0]
v_cmp_lt_u16 [vcc_lo, vcc_hi], v77, s10 src0_sel:WORD_0 src1_sel:WORD_1
v_cndmask_b32 v8, v81, v8, [vcc_lo, vcc_hi]
v_cmp_lt_u16 [vcc_lo, vcc_hi], v88, s10 src0_sel:WORD_0 src1_sel:WORD_1
v_cndmask_b32 v6, v81, v6, [vcc_lo, vcc_hi]
s_branch label_127
label_126:
v_mov_b32 v91, 2147483648
v_mov_b32 v92, s56
v_mov_b32 v81, s64
v_mov_b32 v90, s2
v_add_co_u32 v77, [vcc_lo, vcc_hi], v1, s9
v_mov_b32 v93, 131073
v_pk_mad_u16 v87, v1, v93, s11 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_mad_u16 v88, v1, v93, s14 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_min_u16 v86, v77, v1 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_cndmask_b32 v85, 0, v92, [vcc_lo, vcc_hi]
v_cndmask_b32 v89, 0, v81, [vcc_lo, vcc_hi]
v_mad_u32_u16 v6, v87, 1, v2 op_sel:[0, 0, 0, 0]
v_mad_u32_u16 v12, v88, 1, v11 op_sel:[0, 0, 0, 0]
v_add3_u32 v2, v2, s12, v85
v_add3_u32 v11, v11, s15, v89
v_pk_sub_u16 v86, v1, v86 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_addc_co_u32 v4, [s20, s21], v4, v90, [vcc_lo, vcc_hi]
v_cndmask_b32 v2, v2, v91, [s20, s21]
v_cndmask_b32 v11, v11, v91, [s20, s21]
v_cmp_lt_u16 [vcc_lo, vcc_hi], v87, s10
v_cndmask_b32 v6, v91, v6, [vcc_lo, vcc_hi]
v_cmp_lt_u16 [vcc_lo, vcc_hi], v88, s13
v_cndmask_b32 v12, v91, v12, [vcc_lo, vcc_hi]
v_pk_ashrrev_i16 v86, 15, v86 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_pk_add_u16 v81, v87, s17 op_sel:[1, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_u16 v92, v87, s16 op_sel:[1, 0, 0] op_sel_hi:[1, 1, 1]
v_mad_u32_u16 v9, v81, s58, v6 op_sel:[1, 0, 0, 0]
v_mad_u32_u16 v7, v92, s58, v6 op_sel:[1, 0, 0, 0]
v_pk_add_u16 v90, v88, s18 op_sel:[1, 0, 0] op_sel_hi:[1, 1, 1]
v_cmp_lt_u16 [vcc_lo, vcc_hi], v81, s10 src0_sel:WORD_1 src1_sel:WORD_1
v_cndmask_b32 v9, v91, v9, [vcc_lo, vcc_hi]
v_cmp_lt_u16 [vcc_lo, vcc_hi], v92, s10 src0_sel:WORD_1 src1_sel:WORD_1
v_cndmask_b32 v7, v91, v7, [vcc_lo, vcc_hi]
v_mad_u32_u16 v13, v90, s66, v12 op_sel:[1, 0, 0, 0]
v_mad_u32_u16 v8, v81, s58, v6 op_sel:[0, 0, 0, 0]
v_mad_u32_u16 v6, v92, s58, v6 op_sel:[0, 0, 0, 0]
v_cmp_lt_u16 [vcc_lo, vcc_hi], v90, s13 src0_sel:WORD_1 src1_sel:WORD_1
v_cndmask_b32 v13, v91, v13, [vcc_lo, vcc_hi]
v_cmp_lt_u16 [vcc_lo, vcc_hi], v81, s10 src0_sel:WORD_0 src1_sel:WORD_1
v_cndmask_b32 v8, v91, v8, [vcc_lo, vcc_hi]
v_cmp_lt_u16 [vcc_lo, vcc_hi], v92, s10 src0_sel:WORD_0 src1_sel:WORD_1
v_cndmask_b32 v6, v91, v6, [vcc_lo, vcc_hi]
v_mad_u32_u16 v12, v90, s66, v12 op_sel:[0, 0, 0, 0]
v_pk_mad_u16 v1, v86, s3, v77 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_cmp_lt_u16 [vcc_lo, vcc_hi], v90, s13 src0_sel:WORD_0 src1_sel:WORD_1
v_cndmask_b32 v12, v91, v12, [vcc_lo, vcc_hi]
label_127:
v_and_b32 v84, s73, 3
v_xor_b32 v77, v0, v0 quad_perm:[0, 0, 0, 1]
v_add_u32 v77, s25, v77
v_cmp_lt_u32 [s44, s45], v77, s39
v_mul_i32_i24 v77, s29, v77
s_mul_i32 s41, s19, s24
s_sub_u32 s42, s38, 1
s_sub_u32 s42, s42, s24
s_cselect_b64 [s44, s45], 0, [s44, s45]
s_sub_u32 s42, s42, 1
s_cselect_b64 [s80, s81], 0, [s44, s45]
s_sub_u32 s42, s42, 1
s_cselect_b64 [s82, s83], 0, [s80, s81]
v_mov_b32 v81, 3221225472
v_add3_u32 v85, v77, s41, v5
v_mad_u32_u24 v14, s60, v10, v85
v_cmp_lt_u32 [vcc_lo, vcc_hi], v10, s26
v_cndmask_b32 v14, v81, v14, [vcc_lo, vcc_hi]
v_add_u32 v15, s19, v14
v_add_u32 v16, s19, v15
v_cndmask_b32 v14, v81, v14, [s44, s45]
v_cndmask_b32 v15, v81, v15, [s80, s81]
v_cndmask_b32 v16, v81, v16, [s82, s83]
v_add_u32 v86, v10, 16
v_mad_u32_u24 v18, s60, v86, v85
v_cmp_lt_u32 [vcc_lo, vcc_hi], v86, s26
v_cndmask_b32 v18, v81, v18, [vcc_lo, vcc_hi]
v_add_u32 v19, s19, v18
v_add_u32 v20, s19, v19
v_cndmask_b32 v18, v81, v18, [s44, s45]
v_cndmask_b32 v19, v81, v19, [s80, s81]
v_cndmask_b32 v20, v81, v20, [s82, s83]
ds_write_b128 v75, [v14, v15, v16, v17] offset:45568
ds_write_b128 v75, [v18, v19, v20, v21] offset:46592
ds_write_b128 v74, [v6, v7, v8, v9] offset:53760
ds_write_b32 v76, v84 offset:55808
s_setpc_b64 [s76, s77]
label_97:
label_128:
s_barrier
s_lshl_b32 s73, s73, 2
s_mov_b32 s75, label_130 - label_1
s_sub_u32 s74, s74, 16
s_cbranch_scc1 label_94
label_130:
s_setprio 0
s_setprio 2
s_bitcmp1_b32 s73, 7
s_cselect_b32 s76, s78, label_129 - label_1
s_add_u32 s76, s6, s76
s_addc_u32 s77, s7, 0
s_sub_u32 s50, s50, 1
s_cselect_b32 s50, 2, s50
s_lshl_b32 s51, s50, 9
s_lshl_b32 s52, s50, 2
s_waitcnt lgkmcnt(0)
s_swappc_b64 [s76, s77], [s76, s77]
label_129:
s_bitcmp1_b32 s73, 5
s_cbranch_scc1 label_95
label_96:
s_barrier
s_setprio 0
s_setprio 2
s_bitcmp1_b32 s73, 7
s_cselect_b32 s76, s78, label_131 - label_1
s_add_u32 s76, s6, s76
s_addc_u32 s77, s7, 0
s_waitcnt lgkmcnt(0)
s_swappc_b64 [s76, s77], [s76, s77]
label_131:
s_barrier
s_setprio 0
s_setprio 2
s_bitcmp1_b32 s73, 7
s_cselect_b32 s76, s78, label_132 - label_1
s_add_u32 s76, s6, s76
s_addc_u32 s77, s7, 0
s_waitcnt lgkmcnt(0)
s_swappc_b64 [s76, s77], [s76, s77]
label_132:
s_barrier
s_setprio 0
s_setprio 2
s_bitcmp1_b32 s73, 5
s_cselect_b32 s76, s78, label_133 - label_1
s_add_u32 s76, s6, s76
s_addc_u32 s77, s7, 0
s_waitcnt lgkmcnt(0)
s_swappc_b64 [s76, s77], [s76, s77]
label_133:
s_branch label_128
label_134:
s_barrier
s_lshl_b32 s73, s73, 2
s_mov_b32 s75, label_136 - label_1
s_sub_u32 s74, s74, 16
s_cbranch_scc1 label_94
label_136:
s_setprio 0
ds_read_b64 [v44, v45], v3 offset:8256
ds_read_b64 [v42, v43], v3 offset:8512
ds_read_b64 [v52, v53], v3 offset:8768
ds_read_b64 [v50, v51], v3 offset:9024
s_setprio 3
s_bitcmp1_b32 s73, 7
s_cselect_b32 s76, s78, label_135 - label_1
s_add_u32 s76, s6, s76
s_addc_u32 s77, s7, 0
s_sub_u32 s50, s50, 1
s_cselect_b32 s50, 2, s50
s_lshl_b32 s51, s50, 9
s_lshl_b32 s52, s50, 2
s_waitcnt lgkmcnt(0)
v_fma_mix_f32 v67, v44, 1.0, v42 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 0]
v_fma_mix_f32 v44, v44, 1.0, v42 op_sel:[1, 0, 1, 0] op_sel_hi:[1, 0, 1, 0]
v_fma_mix_f32 v68, v45, 1.0, v43 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 0]
v_fma_mix_f32 v45, v45, 1.0, v43 op_sel:[1, 0, 1, 0] op_sel_hi:[1, 0, 1, 0]
v_fma_mixlo_f16 v67, v67, 1.0, v52 op_sel:[0, 0, 0, 0] op_sel_hi:[0, 0, 1, 0]
v_fma_mixhi_f16 v67, v44, 1.0, v52 op_sel:[0, 0, 1, 0] op_sel_hi:[0, 0, 1, 0]
v_fma_mixlo_f16 v68, v68, 1.0, v53 op_sel:[0, 0, 0, 0] op_sel_hi:[0, 0, 1, 0]
v_fma_mixhi_f16 v68, v45, 1.0, v53 op_sel:[0, 0, 1, 0] op_sel_hi:[0, 0, 1, 0]
v_fma_mix_f32 v69, v52, -1.0, v42 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 0]
v_fma_mix_f32 v52, v52, -1.0, v42 op_sel:[1, 0, 1, 0] op_sel_hi:[1, 0, 1, 0]
v_fma_mix_f32 v70, v53, -1.0, v43 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 0]
v_fma_mix_f32 v53, v53, -1.0, v43 op_sel:[1, 0, 1, 0] op_sel_hi:[1, 0, 1, 0]
v_fma_mixlo_f16 v69, v69, 1.0, v50 op_sel:[0, 0, 0, 0] op_sel_hi:[0, 0, 1, 0]
v_fma_mixhi_f16 v69, v52, 1.0, v50 op_sel:[0, 0, 1, 0] op_sel_hi:[0, 0, 1, 0]
v_fma_mixlo_f16 v70, v70, 1.0, v51 op_sel:[0, 0, 0, 0] op_sel_hi:[0, 0, 1, 0]
v_fma_mixhi_f16 v70, v53, 1.0, v51 op_sel:[0, 0, 1, 0] op_sel_hi:[0, 0, 1, 0]
s_swappc_b64 [s76, s77], [s76, s77]
label_135:
s_bitcmp1_b32 s73, 5
s_cbranch_scc0 label_96
label_95:
s_barrier
s_setprio 0
ds_read_b64 [v24, v25], v3 offset:0
ds_read_b64 [v22, v23], v3 offset:256
ds_read_b64 [v32, v33], v3 offset:512
ds_read_b64 [v30, v31], v3 offset:768
s_setprio 3
s_bitcmp1_b32 s73, 7
s_cselect_b32 s76, s78, label_137 - label_1
s_add_u32 s76, s6, s76
s_addc_u32 s77, s7, 0
s_waitcnt lgkmcnt(0)
s_swappc_b64 [s76, s77], [s76, s77]
label_137:
v_fma_mix_f32 v55, v24, 1.0, v22 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 0]
v_fma_mix_f32 v24, v24, 1.0, v22 op_sel:[1, 0, 1, 0] op_sel_hi:[1, 0, 1, 0]
v_fma_mix_f32 v56, v25, 1.0, v23 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 0]
v_fma_mix_f32 v25, v25, 1.0, v23 op_sel:[1, 0, 1, 0] op_sel_hi:[1, 0, 1, 0]
v_fma_mixlo_f16 v55, v55, 1.0, v32 op_sel:[0, 0, 0, 0] op_sel_hi:[0, 0, 1, 0]
v_fma_mixhi_f16 v55, v24, 1.0, v32 op_sel:[0, 0, 1, 0] op_sel_hi:[0, 0, 1, 0]
v_fma_mixlo_f16 v56, v56, 1.0, v33 op_sel:[0, 0, 0, 0] op_sel_hi:[0, 0, 1, 0]
v_fma_mixhi_f16 v56, v25, 1.0, v33 op_sel:[0, 0, 1, 0] op_sel_hi:[0, 0, 1, 0]
v_fma_mix_f32 v57, v32, -1.0, v22 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 0]
v_fma_mix_f32 v32, v32, -1.0, v22 op_sel:[1, 0, 1, 0] op_sel_hi:[1, 0, 1, 0]
v_fma_mix_f32 v58, v33, -1.0, v23 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 0]
v_fma_mix_f32 v33, v33, -1.0, v23 op_sel:[1, 0, 1, 0] op_sel_hi:[1, 0, 1, 0]
v_fma_mixlo_f16 v57, v57, 1.0, v30 op_sel:[0, 0, 0, 0] op_sel_hi:[0, 0, 1, 0]
v_fma_mixhi_f16 v57, v32, 1.0, v30 op_sel:[0, 0, 1, 0] op_sel_hi:[0, 0, 1, 0]
v_fma_mixlo_f16 v58, v58, 1.0, v31 op_sel:[0, 0, 0, 0] op_sel_hi:[0, 0, 1, 0]
v_fma_mixhi_f16 v58, v33, 1.0, v31 op_sel:[0, 0, 1, 0] op_sel_hi:[0, 0, 1, 0]
s_barrier
s_setprio 0
ds_read_b64 [v40, v41], v3 offset:8256
ds_read_b64 [v38, v39], v3 offset:8512
ds_read_b64 [v48, v49], v3 offset:8768
ds_read_b64 [v46, v47], v3 offset:9024
s_setprio 3
s_bitcmp1_b32 s73, 7
s_cselect_b32 s76, s78, label_138 - label_1
s_add_u32 s76, s6, s76
s_addc_u32 s77, s7, 0
s_waitcnt lgkmcnt(0)
s_swappc_b64 [s76, s77], [s76, s77]
label_138:
v_fma_mix_f32 v63, v40, 1.0, v38 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 0]
v_fma_mix_f32 v40, v40, 1.0, v38 op_sel:[1, 0, 1, 0] op_sel_hi:[1, 0, 1, 0]
v_fma_mix_f32 v64, v41, 1.0, v39 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 0]
v_fma_mix_f32 v41, v41, 1.0, v39 op_sel:[1, 0, 1, 0] op_sel_hi:[1, 0, 1, 0]
v_fma_mixlo_f16 v63, v63, 1.0, v48 op_sel:[0, 0, 0, 0] op_sel_hi:[0, 0, 1, 0]
v_fma_mixhi_f16 v63, v40, 1.0, v48 op_sel:[0, 0, 1, 0] op_sel_hi:[0, 0, 1, 0]
v_fma_mixlo_f16 v64, v64, 1.0, v49 op_sel:[0, 0, 0, 0] op_sel_hi:[0, 0, 1, 0]
v_fma_mixhi_f16 v64, v41, 1.0, v49 op_sel:[0, 0, 1, 0] op_sel_hi:[0, 0, 1, 0]
v_fma_mix_f32 v65, v48, -1.0, v38 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 0]
v_fma_mix_f32 v48, v48, -1.0, v38 op_sel:[1, 0, 1, 0] op_sel_hi:[1, 0, 1, 0]
v_fma_mix_f32 v66, v49, -1.0, v39 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 0]
v_fma_mix_f32 v49, v49, -1.0, v39 op_sel:[1, 0, 1, 0] op_sel_hi:[1, 0, 1, 0]
v_fma_mixlo_f16 v65, v65, 1.0, v46 op_sel:[0, 0, 0, 0] op_sel_hi:[0, 0, 1, 0]
v_fma_mixhi_f16 v65, v48, 1.0, v46 op_sel:[0, 0, 1, 0] op_sel_hi:[0, 0, 1, 0]
v_fma_mixlo_f16 v66, v66, 1.0, v47 op_sel:[0, 0, 0, 0] op_sel_hi:[0, 0, 1, 0]
v_fma_mixhi_f16 v66, v49, 1.0, v47 op_sel:[0, 0, 1, 0] op_sel_hi:[0, 0, 1, 0]
v_add_u32 v83, s51, v71
v_add_u32 v80, s52, v73
s_barrier
s_setprio 0
ds_read_b64 [v78, v79], v83 offset:55816
ds_read_b64 [v82, v83], v83 offset:57352
ds_read_b32 v80, v80 offset:58888
ds_read_b64 [v28, v29], v3 offset:0
ds_read_b64 [v26, v27], v3 offset:256
ds_read_b64 [v36, v37], v3 offset:512
ds_read_b64 [v34, v35], v3 offset:768
s_setprio 3
s_bitcmp1_b32 s73, 5
s_cselect_b32 s76, s78, label_139 - label_1
s_add_u32 s76, s6, s76
s_addc_u32 s77, s7, 0
s_waitcnt lgkmcnt(0)
v_fma_mix_f32 v59, v28, 1.0, v26 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 0]
v_fma_mix_f32 v28, v28, 1.0, v26 op_sel:[1, 0, 1, 0] op_sel_hi:[1, 0, 1, 0]
v_fma_mix_f32 v60, v29, 1.0, v27 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 0]
v_fma_mix_f32 v29, v29, 1.0, v27 op_sel:[1, 0, 1, 0] op_sel_hi:[1, 0, 1, 0]
v_fma_mixlo_f16 v59, v59, 1.0, v36 op_sel:[0, 0, 0, 0] op_sel_hi:[0, 0, 1, 0]
v_fma_mixhi_f16 v59, v28, 1.0, v36 op_sel:[0, 0, 1, 0] op_sel_hi:[0, 0, 1, 0]
v_fma_mixlo_f16 v60, v60, 1.0, v37 op_sel:[0, 0, 0, 0] op_sel_hi:[0, 0, 1, 0]
v_fma_mixhi_f16 v60, v29, 1.0, v37 op_sel:[0, 0, 1, 0] op_sel_hi:[0, 0, 1, 0]
v_fma_mix_f32 v61, v36, -1.0, v26 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 0]
v_fma_mix_f32 v36, v36, -1.0, v26 op_sel:[1, 0, 1, 0] op_sel_hi:[1, 0, 1, 0]
v_fma_mix_f32 v62, v37, -1.0, v27 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 1, 0]
v_fma_mix_f32 v37, v37, -1.0, v27 op_sel:[1, 0, 1, 0] op_sel_hi:[1, 0, 1, 0]
v_fma_mixlo_f16 v61, v61, 1.0, v34 op_sel:[0, 0, 0, 0] op_sel_hi:[0, 0, 1, 0]
v_fma_mixhi_f16 v61, v36, 1.0, v34 op_sel:[0, 0, 1, 0] op_sel_hi:[0, 0, 1, 0]
v_fma_mixlo_f16 v62, v62, 1.0, v35 op_sel:[0, 0, 0, 0] op_sel_hi:[0, 0, 1, 0]
v_fma_mixhi_f16 v62, v37, 1.0, v35 op_sel:[0, 0, 1, 0] op_sel_hi:[0, 0, 1, 0]
s_swappc_b64 [s76, s77], [s76, s77]
label_139:
s_branch label_134
label_98:
s_setprio 3
v_readfirstlane_b32 s53, v80
v_pk_add_f16 v55, v55, s36 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v59, v59, s36 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v57, v57, s36 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v61, v61, s36 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v77, s46, v55 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v87, s46, v59 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_max_f16 v55, v55, v77 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_max_f16 v59, v59, v87 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v77, s46, v57 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v87, s46, v61 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_max_f16 v57, v57, v77 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_max_f16 v61, v61, v87 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
s_setprio 0
buffer_store_short v55, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short v59, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short v57, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short v61, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v55, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v59, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v57, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v61, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s78, label_99 - label_1
s_setpc_b64 [s76, s77]
label_99:
s_setprio 3
v_pk_add_f16 v56, v56, s37 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v60, v60, s37 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v58, v58, s37 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v62, v62, s37 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v85, s46, v56 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v89, s46, v60 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_max_f16 v56, v56, v85 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_max_f16 v60, v60, v89 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v85, s46, v58 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v89, s46, v62 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_max_f16 v58, v58, v85 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_max_f16 v62, v62, v89 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
s_setprio 0
buffer_store_short v56, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short v60, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short v58, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short v62, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v56, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v60, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v58, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v62, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s78, label_100 - label_1
s_sub_u32 s63, s67, 17
s_cselect_b32 s70, 0, s70
s_add_u32 s68, s34, s54
s_addc_u32 s69, s35, s55
s_setpc_b64 [s76, s77]
label_100:
s_setprio 3
v_pk_add_f16 v63, v63, s48 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v67, v67, s48 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v65, v65, s48 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v69, v69, s48 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v77, s46, v63 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v87, s46, v67 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_max_f16 v63, v63, v77 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_max_f16 v67, v67, v87 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v77, s46, v65 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v87, s46, v69 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_max_f16 v65, v65, v77 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_max_f16 v69, v69, v87 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
s_setprio 0
buffer_store_short v63, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short v67, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short v65, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short v69, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v63, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v67, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v65, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v69, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s78, label_101 - label_1
s_setpc_b64 [s76, s77]
label_101:
s_setprio 3
v_pk_add_f16 v64, v64, s49 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v68, v68, s49 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v66, v66, s49 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v70, v70, s49 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v85, s46, v64 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v89, s46, v68 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_max_f16 v64, v64, v85 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_max_f16 v68, v68, v89 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v85, s46, v66 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v89, s46, v70 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_max_f16 v66, v66, v85 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_max_f16 v70, v70, v89 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
s_setprio 0
buffer_store_short v64, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short v68, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short v66, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short v70, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v64, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v68, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v66, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v70, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s78, label_98 - label_1
s_bitcmp1_b32 s53, 0
s_cbranch_scc1 label_0
s_sub_u32 s63, s67, 1
s_cselect_b32 s70, 0, 2147483648
s_mov_b64 [s68, s69], [s34, s35]
s_setpc_b64 [s76, s77]
label_102:
s_setprio 3
v_readfirstlane_b32 s53, v80
v_pk_add_f16 v55, v55, s36 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v59, v59, s36 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v57, v57, s36 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v61, v61, s36 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v77, s46, v55 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v87, s46, v59 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_min_f16 v55, v55, v77 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_min_f16 v59, v59, v87 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v77, s46, v57 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v87, s46, v61 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_min_f16 v57, v57, v77 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_min_f16 v61, v61, v87 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
s_setprio 0
buffer_store_short v55, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short v59, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short v57, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short v61, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v55, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v59, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v57, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v61, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s78, label_103 - label_1
s_setpc_b64 [s76, s77]
label_103:
s_setprio 3
v_pk_add_f16 v56, v56, s37 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v60, v60, s37 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v58, v58, s37 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v62, v62, s37 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v85, s46, v56 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v89, s46, v60 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_min_f16 v56, v56, v85 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_min_f16 v60, v60, v89 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v85, s46, v58 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v89, s46, v62 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_min_f16 v58, v58, v85 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_min_f16 v62, v62, v89 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
s_setprio 0
buffer_store_short v56, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short v60, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short v58, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short v62, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v56, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v60, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v58, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v62, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s78, label_104 - label_1
s_sub_u32 s63, s67, 17
s_cselect_b32 s70, 0, s70
s_add_u32 s68, s34, s54
s_addc_u32 s69, s35, s55
s_setpc_b64 [s76, s77]
label_104:
s_setprio 3
v_pk_add_f16 v63, v63, s48 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v67, v67, s48 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v65, v65, s48 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v69, v69, s48 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v77, s46, v63 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v87, s46, v67 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_min_f16 v63, v63, v77 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_min_f16 v67, v67, v87 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v77, s46, v65 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v87, s46, v69 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_min_f16 v65, v65, v77 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_min_f16 v69, v69, v87 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
s_setprio 0
buffer_store_short v63, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short v67, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short v65, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short v69, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v63, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v67, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v65, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v69, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s78, label_105 - label_1
s_setpc_b64 [s76, s77]
label_105:
s_setprio 3
v_pk_add_f16 v64, v64, s49 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v68, v68, s49 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v66, v66, s49 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v70, v70, s49 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v85, s46, v64 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v89, s46, v68 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_min_f16 v64, v64, v85 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_min_f16 v68, v68, v89 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v85, s46, v66 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v89, s46, v70 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_min_f16 v66, v66, v85 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_min_f16 v70, v70, v89 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
s_setprio 0
buffer_store_short v64, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short v68, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short v66, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short v70, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v64, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v68, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v66, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v70, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s78, label_102 - label_1
s_bitcmp1_b32 s53, 0
s_cbranch_scc1 label_0
s_sub_u32 s63, s67, 1
s_cselect_b32 s70, 0, 2147483648
s_mov_b64 [s68, s69], [s34, s35]
s_setpc_b64 [s76, s77]
label_106:
s_setprio 3
v_readfirstlane_b32 s53, v80
v_pk_add_f16 v55, v55, s36 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v59, v59, s36 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v57, v57, s36 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v61, v61, s36 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_max_f16 v55, 0, v55 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_max_f16 v59, 0, v59 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_max_f16 v57, 0, v57 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_max_f16 v61, 0, v61 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
s_setprio 0
buffer_store_short v55, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short v59, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short v57, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short v61, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v55, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v59, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v57, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v61, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s78, label_107 - label_1
s_setpc_b64 [s76, s77]
label_107:
s_setprio 3
v_pk_add_f16 v56, v56, s37 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v60, v60, s37 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v58, v58, s37 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v62, v62, s37 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_max_f16 v56, 0, v56 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_max_f16 v60, 0, v60 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_max_f16 v58, 0, v58 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_max_f16 v62, 0, v62 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
s_setprio 0
buffer_store_short v56, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short v60, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short v58, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short v62, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v56, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v60, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v58, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v62, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s78, label_108 - label_1
s_sub_u32 s63, s67, 17
s_cselect_b32 s70, 0, s70
s_add_u32 s68, s34, s54
s_addc_u32 s69, s35, s55
s_setpc_b64 [s76, s77]
label_108:
s_setprio 3
v_pk_add_f16 v63, v63, s48 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v67, v67, s48 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v65, v65, s48 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v69, v69, s48 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_max_f16 v63, 0, v63 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_max_f16 v67, 0, v67 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_max_f16 v65, 0, v65 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_max_f16 v69, 0, v69 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
s_setprio 0
buffer_store_short v63, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short v67, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short v65, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short v69, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v63, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v67, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v65, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v69, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s78, label_109 - label_1
s_setpc_b64 [s76, s77]
label_109:
s_setprio 3
v_pk_add_f16 v64, v64, s49 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v68, v68, s49 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v66, v66, s49 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v70, v70, s49 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_max_f16 v64, 0, v64 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_max_f16 v68, 0, v68 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_max_f16 v66, 0, v66 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_max_f16 v70, 0, v70 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
s_setprio 0
buffer_store_short v64, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short v68, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short v66, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short v70, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v64, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v68, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v66, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v70, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s78, label_106 - label_1
s_bitcmp1_b32 s53, 0
s_cbranch_scc1 label_0
s_sub_u32 s63, s67, 1
s_cselect_b32 s70, 0, 2147483648
s_mov_b64 [s68, s69], [s34, s35]
s_setpc_b64 [s76, s77]
label_110:
s_setprio 3
v_readfirstlane_b32 s53, v80
v_pk_add_f16 v55, v55, s36 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v59, v59, s36 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v57, v57, s36 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v61, v61, s36 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mov_b32 v91, 3183852997
v_pk_mul_f16 v55, v91, v55 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mov_b32 v91, 3183852997
v_pk_mul_f16 v59, v91, v59 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_exp_f16 v55, v55 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_exp_f16 v59, v59 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_nop
v_nop
v_exp_f16 v55, v55 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_exp_f16 v59, v59 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_nop
v_nop
v_pk_add_f16 v55, 1.0, v55 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_pk_add_f16 v59, 1.0, v59 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_rcp_f16 v55, v55 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_rcp_f16 v59, v59 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_nop
v_nop
v_rcp_f16 v55, v55 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_rcp_f16 v59, v59 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_nop
v_nop
v_mov_b32 v91, 3183852997
v_pk_mul_f16 v57, v91, v57 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mov_b32 v91, 3183852997
v_pk_mul_f16 v61, v91, v61 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_exp_f16 v57, v57 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_exp_f16 v61, v61 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_nop
v_nop
v_exp_f16 v57, v57 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_exp_f16 v61, v61 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_nop
v_nop
v_pk_add_f16 v57, 1.0, v57 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_pk_add_f16 v61, 1.0, v61 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_rcp_f16 v57, v57 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_rcp_f16 v61, v61 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_nop
v_nop
v_rcp_f16 v57, v57 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_rcp_f16 v61, v61 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_nop
v_nop
s_setprio 0
buffer_store_short v55, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short v59, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short v57, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short v61, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v55, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v59, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v57, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v61, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s78, label_111 - label_1
s_setpc_b64 [s76, s77]
label_111:
s_setprio 3
v_pk_add_f16 v56, v56, s37 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v60, v60, s37 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v58, v58, s37 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v62, v62, s37 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mov_b32 v91, 3183852997
v_pk_mul_f16 v56, v91, v56 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mov_b32 v91, 3183852997
v_pk_mul_f16 v60, v91, v60 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_exp_f16 v56, v56 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_exp_f16 v60, v60 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_nop
v_nop
v_exp_f16 v56, v56 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_exp_f16 v60, v60 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_nop
v_nop
v_pk_add_f16 v56, 1.0, v56 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_pk_add_f16 v60, 1.0, v60 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_rcp_f16 v56, v56 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_rcp_f16 v60, v60 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_nop
v_nop
v_rcp_f16 v56, v56 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_rcp_f16 v60, v60 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_nop
v_nop
v_mov_b32 v91, 3183852997
v_pk_mul_f16 v58, v91, v58 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mov_b32 v91, 3183852997
v_pk_mul_f16 v62, v91, v62 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_exp_f16 v58, v58 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_exp_f16 v62, v62 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_nop
v_nop
v_exp_f16 v58, v58 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_exp_f16 v62, v62 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_nop
v_nop
v_pk_add_f16 v58, 1.0, v58 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_pk_add_f16 v62, 1.0, v62 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_rcp_f16 v58, v58 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_rcp_f16 v62, v62 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_nop
v_nop
v_rcp_f16 v58, v58 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_rcp_f16 v62, v62 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_nop
v_nop
s_setprio 0
buffer_store_short v56, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short v60, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short v58, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short v62, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v56, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v60, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v58, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v62, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s78, label_112 - label_1
s_sub_u32 s63, s67, 17
s_cselect_b32 s70, 0, s70
s_add_u32 s68, s34, s54
s_addc_u32 s69, s35, s55
s_setpc_b64 [s76, s77]
label_112:
s_setprio 3
v_pk_add_f16 v63, v63, s48 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v67, v67, s48 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v65, v65, s48 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v69, v69, s48 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mov_b32 v91, 3183852997
v_pk_mul_f16 v63, v91, v63 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mov_b32 v91, 3183852997
v_pk_mul_f16 v67, v91, v67 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_exp_f16 v63, v63 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_exp_f16 v67, v67 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_nop
v_nop
v_exp_f16 v63, v63 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_exp_f16 v67, v67 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_nop
v_nop
v_pk_add_f16 v63, 1.0, v63 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_pk_add_f16 v67, 1.0, v67 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_rcp_f16 v63, v63 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_rcp_f16 v67, v67 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_nop
v_nop
v_rcp_f16 v63, v63 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_rcp_f16 v67, v67 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_nop
v_nop
v_mov_b32 v91, 3183852997
v_pk_mul_f16 v65, v91, v65 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mov_b32 v91, 3183852997
v_pk_mul_f16 v69, v91, v69 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_exp_f16 v65, v65 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_exp_f16 v69, v69 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_nop
v_nop
v_exp_f16 v65, v65 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_exp_f16 v69, v69 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_nop
v_nop
v_pk_add_f16 v65, 1.0, v65 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_pk_add_f16 v69, 1.0, v69 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_rcp_f16 v65, v65 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_rcp_f16 v69, v69 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_nop
v_nop
v_rcp_f16 v65, v65 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_rcp_f16 v69, v69 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_nop
v_nop
s_setprio 0
buffer_store_short v63, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short v67, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short v65, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short v69, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v63, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v67, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v65, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v69, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s78, label_113 - label_1
s_setpc_b64 [s76, s77]
label_113:
s_setprio 3
v_pk_add_f16 v64, v64, s49 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v68, v68, s49 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v66, v66, s49 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v70, v70, s49 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mov_b32 v91, 3183852997
v_pk_mul_f16 v64, v91, v64 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mov_b32 v91, 3183852997
v_pk_mul_f16 v68, v91, v68 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_exp_f16 v64, v64 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_exp_f16 v68, v68 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_nop
v_nop
v_exp_f16 v64, v64 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_exp_f16 v68, v68 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_nop
v_nop
v_pk_add_f16 v64, 1.0, v64 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_pk_add_f16 v68, 1.0, v68 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_rcp_f16 v64, v64 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_rcp_f16 v68, v68 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_nop
v_nop
v_rcp_f16 v64, v64 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_rcp_f16 v68, v68 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_nop
v_nop
v_mov_b32 v91, 3183852997
v_pk_mul_f16 v66, v91, v66 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mov_b32 v91, 3183852997
v_pk_mul_f16 v70, v91, v70 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_exp_f16 v66, v66 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_exp_f16 v70, v70 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_nop
v_nop
v_exp_f16 v66, v66 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_exp_f16 v70, v70 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_nop
v_nop
v_pk_add_f16 v66, 1.0, v66 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_pk_add_f16 v70, 1.0, v70 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_rcp_f16 v66, v66 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_rcp_f16 v70, v70 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_nop
v_nop
v_rcp_f16 v66, v66 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_rcp_f16 v70, v70 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_nop
v_nop
s_setprio 0
buffer_store_short v64, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short v68, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short v66, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short v70, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v64, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v68, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v66, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v70, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s78, label_110 - label_1
s_bitcmp1_b32 s53, 0
s_cbranch_scc1 label_0
s_sub_u32 s63, s67, 1
s_cselect_b32 s70, 0, 2147483648
s_mov_b64 [s68, s69], [s34, s35]
s_setpc_b64 [s76, s77]
label_114:
s_setprio 3
v_readfirstlane_b32 s53, v80
v_pk_add_f16 v55, v55, s36 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v59, v59, s36 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v57, v57, s36 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v61, v61, s36 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v55, v55, s47 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v59, v59, s47 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_and_b32 v77, 2147450879, v55
v_and_b32 v87, 2147450879, v59
v_mov_b32 v81, 3056514606
v_mov_b32 v88, 3056514606
v_mov_b32 v91, 814231688
v_pk_fma_f16 v81, v77, v91, v81 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_mov_b32 v91, 814231688
v_pk_fma_f16 v88, v87, v91, v88 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_mov_b32 v91, 506666547
v_pk_fma_f16 v81, v77, v81, v91 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_mov_b32 v91, 506666547
v_pk_fma_f16 v88, v87, v88, v91 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_mul_f16 v81, v77, v81 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v88, v87, v88 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f16 v81, v77, v81, v77 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f16 v88, v87, v88, v87 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_mov_b32 v91, 1103446469
v_pk_mul_f16 v77, v91, v77 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mov_b32 v91, 1103446469
v_pk_mul_f16 v87, v91, v87 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_exp_f16 v77, v77 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_exp_f16 v87, v87 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_nop
v_nop
v_exp_f16 v77, v77 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_exp_f16 v87, v87 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_nop
v_nop
v_pk_add_f16 v77, 1.0, v77 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_pk_add_f16 v87, 1.0, v87 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_rcp_f16 v77, v77 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_rcp_f16 v87, v87 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_nop
v_nop
v_rcp_f16 v77, v77 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_rcp_f16 v87, v87 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_nop
v_nop
v_pk_fma_f16 v77, v77, -2.0, 1.0 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 0, 1]
v_pk_fma_f16 v87, v87, -2.0, 1.0 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 0, 1]
v_mov_b32 v91, 1002585026
v_cmp_gt_f16 [vcc_lo, vcc_hi], abs(v55), v91
v_cndmask_b32 v81, v81, v77, [vcc_lo, vcc_hi] dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 src1_sel:WORD_0
v_mov_b32 v91, 1002585026
v_cmp_gt_f16 [vcc_lo, vcc_hi], abs(v59), v91
v_cndmask_b32 v88, v88, v87, [vcc_lo, vcc_hi] dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 src1_sel:WORD_0
v_mov_b32 v91, 1002585026
v_cmp_gt_f16 [vcc_lo, vcc_hi], abs(v55), v91 src0_sel:WORD_1 src1_sel:WORD_1
v_cndmask_b32 v81, v81, v77, [vcc_lo, vcc_hi] dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_1
v_mov_b32 v91, 1002585026
v_cmp_gt_f16 [vcc_lo, vcc_hi], abs(v59), v91 src0_sel:WORD_1 src1_sel:WORD_1
v_cndmask_b32 v88, v88, v87, [vcc_lo, vcc_hi] dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_1
v_mov_b32 v77, 2147450879
v_bfi_b32 v55, v77, v81, v55
v_mov_b32 v87, 2147450879
v_bfi_b32 v59, v87, v88, v59
v_pk_mul_f16 v55, v55, s46 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v59, v59, s46 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v57, v57, s47 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v61, v61, s47 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_and_b32 v77, 2147450879, v57
v_and_b32 v87, 2147450879, v61
v_mov_b32 v81, 3056514606
v_mov_b32 v88, 3056514606
v_mov_b32 v91, 814231688
v_pk_fma_f16 v81, v77, v91, v81 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_mov_b32 v91, 814231688
v_pk_fma_f16 v88, v87, v91, v88 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_mov_b32 v91, 506666547
v_pk_fma_f16 v81, v77, v81, v91 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_mov_b32 v91, 506666547
v_pk_fma_f16 v88, v87, v88, v91 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_mul_f16 v81, v77, v81 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v88, v87, v88 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f16 v81, v77, v81, v77 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f16 v88, v87, v88, v87 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_mov_b32 v91, 1103446469
v_pk_mul_f16 v77, v91, v77 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mov_b32 v91, 1103446469
v_pk_mul_f16 v87, v91, v87 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_exp_f16 v77, v77 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_exp_f16 v87, v87 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_nop
v_nop
v_exp_f16 v77, v77 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_exp_f16 v87, v87 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_nop
v_nop
v_pk_add_f16 v77, 1.0, v77 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_pk_add_f16 v87, 1.0, v87 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_rcp_f16 v77, v77 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_rcp_f16 v87, v87 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_nop
v_nop
v_rcp_f16 v77, v77 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_rcp_f16 v87, v87 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_nop
v_nop
v_pk_fma_f16 v77, v77, -2.0, 1.0 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 0, 1]
v_pk_fma_f16 v87, v87, -2.0, 1.0 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 0, 1]
v_mov_b32 v91, 1002585026
v_cmp_gt_f16 [vcc_lo, vcc_hi], abs(v57), v91
v_cndmask_b32 v81, v81, v77, [vcc_lo, vcc_hi] dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 src1_sel:WORD_0
v_mov_b32 v91, 1002585026
v_cmp_gt_f16 [vcc_lo, vcc_hi], abs(v61), v91
v_cndmask_b32 v88, v88, v87, [vcc_lo, vcc_hi] dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 src1_sel:WORD_0
v_mov_b32 v91, 1002585026
v_cmp_gt_f16 [vcc_lo, vcc_hi], abs(v57), v91 src0_sel:WORD_1 src1_sel:WORD_1
v_cndmask_b32 v81, v81, v77, [vcc_lo, vcc_hi] dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_1
v_mov_b32 v91, 1002585026
v_cmp_gt_f16 [vcc_lo, vcc_hi], abs(v61), v91 src0_sel:WORD_1 src1_sel:WORD_1
v_cndmask_b32 v88, v88, v87, [vcc_lo, vcc_hi] dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_1
v_mov_b32 v77, 2147450879
v_bfi_b32 v57, v77, v81, v57
v_mov_b32 v87, 2147450879
v_bfi_b32 v61, v87, v88, v61
v_pk_mul_f16 v57, v57, s46 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v61, v61, s46 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
s_setprio 0
buffer_store_short v55, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short v59, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short v57, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short v61, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v55, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v59, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v57, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v61, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s78, label_115 - label_1
s_setpc_b64 [s76, s77]
label_115:
s_setprio 3
v_pk_add_f16 v56, v56, s37 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v60, v60, s37 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v58, v58, s37 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v62, v62, s37 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v56, v56, s47 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v60, v60, s47 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_and_b32 v85, 2147450879, v56
v_and_b32 v89, 2147450879, v60
v_mov_b32 v86, 3056514606
v_mov_b32 v90, 3056514606
v_mov_b32 v91, 814231688
v_pk_fma_f16 v86, v85, v91, v86 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_mov_b32 v91, 814231688
v_pk_fma_f16 v90, v89, v91, v90 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_mov_b32 v91, 506666547
v_pk_fma_f16 v86, v85, v86, v91 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_mov_b32 v91, 506666547
v_pk_fma_f16 v90, v89, v90, v91 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_mul_f16 v86, v85, v86 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v90, v89, v90 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f16 v86, v85, v86, v85 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f16 v90, v89, v90, v89 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_mov_b32 v91, 1103446469
v_pk_mul_f16 v85, v91, v85 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mov_b32 v91, 1103446469
v_pk_mul_f16 v89, v91, v89 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_exp_f16 v85, v85 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_exp_f16 v89, v89 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_nop
v_nop
v_exp_f16 v85, v85 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_exp_f16 v89, v89 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_nop
v_nop
v_pk_add_f16 v85, 1.0, v85 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_pk_add_f16 v89, 1.0, v89 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_rcp_f16 v85, v85 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_rcp_f16 v89, v89 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_nop
v_nop
v_rcp_f16 v85, v85 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_rcp_f16 v89, v89 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_nop
v_nop
v_pk_fma_f16 v85, v85, -2.0, 1.0 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 0, 1]
v_pk_fma_f16 v89, v89, -2.0, 1.0 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 0, 1]
v_mov_b32 v91, 1002585026
v_cmp_gt_f16 [vcc_lo, vcc_hi], abs(v56), v91
v_cndmask_b32 v86, v86, v85, [vcc_lo, vcc_hi] dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 src1_sel:WORD_0
v_mov_b32 v91, 1002585026
v_cmp_gt_f16 [vcc_lo, vcc_hi], abs(v60), v91
v_cndmask_b32 v90, v90, v89, [vcc_lo, vcc_hi] dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 src1_sel:WORD_0
v_mov_b32 v91, 1002585026
v_cmp_gt_f16 [vcc_lo, vcc_hi], abs(v56), v91 src0_sel:WORD_1 src1_sel:WORD_1
v_cndmask_b32 v86, v86, v85, [vcc_lo, vcc_hi] dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_1
v_mov_b32 v91, 1002585026
v_cmp_gt_f16 [vcc_lo, vcc_hi], abs(v60), v91 src0_sel:WORD_1 src1_sel:WORD_1
v_cndmask_b32 v90, v90, v89, [vcc_lo, vcc_hi] dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_1
v_mov_b32 v85, 2147450879
v_bfi_b32 v56, v85, v86, v56
v_mov_b32 v89, 2147450879
v_bfi_b32 v60, v89, v90, v60
v_pk_mul_f16 v56, v56, s46 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v60, v60, s46 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v58, v58, s47 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v62, v62, s47 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_and_b32 v85, 2147450879, v58
v_and_b32 v89, 2147450879, v62
v_mov_b32 v86, 3056514606
v_mov_b32 v90, 3056514606
v_mov_b32 v91, 814231688
v_pk_fma_f16 v86, v85, v91, v86 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_mov_b32 v91, 814231688
v_pk_fma_f16 v90, v89, v91, v90 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_mov_b32 v91, 506666547
v_pk_fma_f16 v86, v85, v86, v91 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_mov_b32 v91, 506666547
v_pk_fma_f16 v90, v89, v90, v91 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_mul_f16 v86, v85, v86 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v90, v89, v90 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f16 v86, v85, v86, v85 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f16 v90, v89, v90, v89 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_mov_b32 v91, 1103446469
v_pk_mul_f16 v85, v91, v85 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mov_b32 v91, 1103446469
v_pk_mul_f16 v89, v91, v89 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_exp_f16 v85, v85 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_exp_f16 v89, v89 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_nop
v_nop
v_exp_f16 v85, v85 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_exp_f16 v89, v89 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_nop
v_nop
v_pk_add_f16 v85, 1.0, v85 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_pk_add_f16 v89, 1.0, v89 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_rcp_f16 v85, v85 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_rcp_f16 v89, v89 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_nop
v_nop
v_rcp_f16 v85, v85 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_rcp_f16 v89, v89 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_nop
v_nop
v_pk_fma_f16 v85, v85, -2.0, 1.0 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 0, 1]
v_pk_fma_f16 v89, v89, -2.0, 1.0 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 0, 1]
v_mov_b32 v91, 1002585026
v_cmp_gt_f16 [vcc_lo, vcc_hi], abs(v58), v91
v_cndmask_b32 v86, v86, v85, [vcc_lo, vcc_hi] dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 src1_sel:WORD_0
v_mov_b32 v91, 1002585026
v_cmp_gt_f16 [vcc_lo, vcc_hi], abs(v62), v91
v_cndmask_b32 v90, v90, v89, [vcc_lo, vcc_hi] dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 src1_sel:WORD_0
v_mov_b32 v91, 1002585026
v_cmp_gt_f16 [vcc_lo, vcc_hi], abs(v58), v91 src0_sel:WORD_1 src1_sel:WORD_1
v_cndmask_b32 v86, v86, v85, [vcc_lo, vcc_hi] dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_1
v_mov_b32 v91, 1002585026
v_cmp_gt_f16 [vcc_lo, vcc_hi], abs(v62), v91 src0_sel:WORD_1 src1_sel:WORD_1
v_cndmask_b32 v90, v90, v89, [vcc_lo, vcc_hi] dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_1
v_mov_b32 v85, 2147450879
v_bfi_b32 v58, v85, v86, v58
v_mov_b32 v89, 2147450879
v_bfi_b32 v62, v89, v90, v62
v_pk_mul_f16 v58, v58, s46 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v62, v62, s46 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
s_setprio 0
buffer_store_short v56, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short v60, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short v58, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short v62, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v56, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v60, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v58, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v62, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s78, label_116 - label_1
s_sub_u32 s63, s67, 17
s_cselect_b32 s70, 0, s70
s_add_u32 s68, s34, s54
s_addc_u32 s69, s35, s55
s_setpc_b64 [s76, s77]
label_116:
s_setprio 3
v_pk_add_f16 v63, v63, s48 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v67, v67, s48 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v65, v65, s48 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v69, v69, s48 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v63, v63, s47 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v67, v67, s47 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_and_b32 v77, 2147450879, v63
v_and_b32 v87, 2147450879, v67
v_mov_b32 v81, 3056514606
v_mov_b32 v88, 3056514606
v_mov_b32 v91, 814231688
v_pk_fma_f16 v81, v77, v91, v81 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_mov_b32 v91, 814231688
v_pk_fma_f16 v88, v87, v91, v88 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_mov_b32 v91, 506666547
v_pk_fma_f16 v81, v77, v81, v91 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_mov_b32 v91, 506666547
v_pk_fma_f16 v88, v87, v88, v91 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_mul_f16 v81, v77, v81 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v88, v87, v88 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f16 v81, v77, v81, v77 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f16 v88, v87, v88, v87 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_mov_b32 v91, 1103446469
v_pk_mul_f16 v77, v91, v77 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mov_b32 v91, 1103446469
v_pk_mul_f16 v87, v91, v87 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_exp_f16 v77, v77 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_exp_f16 v87, v87 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_nop
v_nop
v_exp_f16 v77, v77 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_exp_f16 v87, v87 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_nop
v_nop
v_pk_add_f16 v77, 1.0, v77 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_pk_add_f16 v87, 1.0, v87 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_rcp_f16 v77, v77 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_rcp_f16 v87, v87 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_nop
v_nop
v_rcp_f16 v77, v77 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_rcp_f16 v87, v87 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_nop
v_nop
v_pk_fma_f16 v77, v77, -2.0, 1.0 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 0, 1]
v_pk_fma_f16 v87, v87, -2.0, 1.0 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 0, 1]
v_mov_b32 v91, 1002585026
v_cmp_gt_f16 [vcc_lo, vcc_hi], abs(v63), v91
v_cndmask_b32 v81, v81, v77, [vcc_lo, vcc_hi] dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 src1_sel:WORD_0
v_mov_b32 v91, 1002585026
v_cmp_gt_f16 [vcc_lo, vcc_hi], abs(v67), v91
v_cndmask_b32 v88, v88, v87, [vcc_lo, vcc_hi] dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 src1_sel:WORD_0
v_mov_b32 v91, 1002585026
v_cmp_gt_f16 [vcc_lo, vcc_hi], abs(v63), v91 src0_sel:WORD_1 src1_sel:WORD_1
v_cndmask_b32 v81, v81, v77, [vcc_lo, vcc_hi] dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_1
v_mov_b32 v91, 1002585026
v_cmp_gt_f16 [vcc_lo, vcc_hi], abs(v67), v91 src0_sel:WORD_1 src1_sel:WORD_1
v_cndmask_b32 v88, v88, v87, [vcc_lo, vcc_hi] dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_1
v_mov_b32 v77, 2147450879
v_bfi_b32 v63, v77, v81, v63
v_mov_b32 v87, 2147450879
v_bfi_b32 v67, v87, v88, v67
v_pk_mul_f16 v63, v63, s46 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v67, v67, s46 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v65, v65, s47 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v69, v69, s47 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_and_b32 v77, 2147450879, v65
v_and_b32 v87, 2147450879, v69
v_mov_b32 v81, 3056514606
v_mov_b32 v88, 3056514606
v_mov_b32 v91, 814231688
v_pk_fma_f16 v81, v77, v91, v81 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_mov_b32 v91, 814231688
v_pk_fma_f16 v88, v87, v91, v88 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_mov_b32 v91, 506666547
v_pk_fma_f16 v81, v77, v81, v91 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_mov_b32 v91, 506666547
v_pk_fma_f16 v88, v87, v88, v91 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_mul_f16 v81, v77, v81 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v88, v87, v88 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f16 v81, v77, v81, v77 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f16 v88, v87, v88, v87 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_mov_b32 v91, 1103446469
v_pk_mul_f16 v77, v91, v77 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mov_b32 v91, 1103446469
v_pk_mul_f16 v87, v91, v87 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_exp_f16 v77, v77 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_exp_f16 v87, v87 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_nop
v_nop
v_exp_f16 v77, v77 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_exp_f16 v87, v87 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_nop
v_nop
v_pk_add_f16 v77, 1.0, v77 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_pk_add_f16 v87, 1.0, v87 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_rcp_f16 v77, v77 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_rcp_f16 v87, v87 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_nop
v_nop
v_rcp_f16 v77, v77 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_rcp_f16 v87, v87 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_nop
v_nop
v_pk_fma_f16 v77, v77, -2.0, 1.0 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 0, 1]
v_pk_fma_f16 v87, v87, -2.0, 1.0 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 0, 1]
v_mov_b32 v91, 1002585026
v_cmp_gt_f16 [vcc_lo, vcc_hi], abs(v65), v91
v_cndmask_b32 v81, v81, v77, [vcc_lo, vcc_hi] dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 src1_sel:WORD_0
v_mov_b32 v91, 1002585026
v_cmp_gt_f16 [vcc_lo, vcc_hi], abs(v69), v91
v_cndmask_b32 v88, v88, v87, [vcc_lo, vcc_hi] dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 src1_sel:WORD_0
v_mov_b32 v91, 1002585026
v_cmp_gt_f16 [vcc_lo, vcc_hi], abs(v65), v91 src0_sel:WORD_1 src1_sel:WORD_1
v_cndmask_b32 v81, v81, v77, [vcc_lo, vcc_hi] dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_1
v_mov_b32 v91, 1002585026
v_cmp_gt_f16 [vcc_lo, vcc_hi], abs(v69), v91 src0_sel:WORD_1 src1_sel:WORD_1
v_cndmask_b32 v88, v88, v87, [vcc_lo, vcc_hi] dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_1
v_mov_b32 v77, 2147450879
v_bfi_b32 v65, v77, v81, v65
v_mov_b32 v87, 2147450879
v_bfi_b32 v69, v87, v88, v69
v_pk_mul_f16 v65, v65, s46 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v69, v69, s46 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
s_setprio 0
buffer_store_short v63, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short v67, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short v65, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short v69, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v63, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v67, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v65, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v69, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s78, label_117 - label_1
s_setpc_b64 [s76, s77]
label_117:
s_setprio 3
v_pk_add_f16 v64, v64, s49 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v68, v68, s49 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v66, v66, s49 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f16 v70, v70, s49 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v64, v64, s47 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v68, v68, s47 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_and_b32 v85, 2147450879, v64
v_and_b32 v89, 2147450879, v68
v_mov_b32 v86, 3056514606
v_mov_b32 v90, 3056514606
v_mov_b32 v91, 814231688
v_pk_fma_f16 v86, v85, v91, v86 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_mov_b32 v91, 814231688
v_pk_fma_f16 v90, v89, v91, v90 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_mov_b32 v91, 506666547
v_pk_fma_f16 v86, v85, v86, v91 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_mov_b32 v91, 506666547
v_pk_fma_f16 v90, v89, v90, v91 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_mul_f16 v86, v85, v86 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v90, v89, v90 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f16 v86, v85, v86, v85 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f16 v90, v89, v90, v89 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_mov_b32 v91, 1103446469
v_pk_mul_f16 v85, v91, v85 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mov_b32 v91, 1103446469
v_pk_mul_f16 v89, v91, v89 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_exp_f16 v85, v85 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_exp_f16 v89, v89 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_nop
v_nop
v_exp_f16 v85, v85 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_exp_f16 v89, v89 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_nop
v_nop
v_pk_add_f16 v85, 1.0, v85 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_pk_add_f16 v89, 1.0, v89 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_rcp_f16 v85, v85 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_rcp_f16 v89, v89 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_nop
v_nop
v_rcp_f16 v85, v85 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_rcp_f16 v89, v89 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_nop
v_nop
v_pk_fma_f16 v85, v85, -2.0, 1.0 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 0, 1]
v_pk_fma_f16 v89, v89, -2.0, 1.0 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 0, 1]
v_mov_b32 v91, 1002585026
v_cmp_gt_f16 [vcc_lo, vcc_hi], abs(v64), v91
v_cndmask_b32 v86, v86, v85, [vcc_lo, vcc_hi] dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 src1_sel:WORD_0
v_mov_b32 v91, 1002585026
v_cmp_gt_f16 [vcc_lo, vcc_hi], abs(v68), v91
v_cndmask_b32 v90, v90, v89, [vcc_lo, vcc_hi] dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 src1_sel:WORD_0
v_mov_b32 v91, 1002585026
v_cmp_gt_f16 [vcc_lo, vcc_hi], abs(v64), v91 src0_sel:WORD_1 src1_sel:WORD_1
v_cndmask_b32 v86, v86, v85, [vcc_lo, vcc_hi] dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_1
v_mov_b32 v91, 1002585026
v_cmp_gt_f16 [vcc_lo, vcc_hi], abs(v68), v91 src0_sel:WORD_1 src1_sel:WORD_1
v_cndmask_b32 v90, v90, v89, [vcc_lo, vcc_hi] dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_1
v_mov_b32 v85, 2147450879
v_bfi_b32 v64, v85, v86, v64
v_mov_b32 v89, 2147450879
v_bfi_b32 v68, v89, v90, v68
v_pk_mul_f16 v64, v64, s46 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v68, v68, s46 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v66, v66, s47 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v70, v70, s47 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_and_b32 v85, 2147450879, v66
v_and_b32 v89, 2147450879, v70
v_mov_b32 v86, 3056514606
v_mov_b32 v90, 3056514606
v_mov_b32 v91, 814231688
v_pk_fma_f16 v86, v85, v91, v86 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_mov_b32 v91, 814231688
v_pk_fma_f16 v90, v89, v91, v90 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_mov_b32 v91, 506666547
v_pk_fma_f16 v86, v85, v86, v91 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_mov_b32 v91, 506666547
v_pk_fma_f16 v90, v89, v90, v91 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_mul_f16 v86, v85, v86 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v90, v89, v90 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f16 v86, v85, v86, v85 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f16 v90, v89, v90, v89 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_mov_b32 v91, 1103446469
v_pk_mul_f16 v85, v91, v85 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mov_b32 v91, 1103446469
v_pk_mul_f16 v89, v91, v89 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_exp_f16 v85, v85 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_exp_f16 v89, v89 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_nop
v_nop
v_exp_f16 v85, v85 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_exp_f16 v89, v89 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_nop
v_nop
v_pk_add_f16 v85, 1.0, v85 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_pk_add_f16 v89, 1.0, v89 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_rcp_f16 v85, v85 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_rcp_f16 v89, v89 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
v_nop
v_nop
v_rcp_f16 v85, v85 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_rcp_f16 v89, v89 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
v_nop
v_nop
v_pk_fma_f16 v85, v85, -2.0, 1.0 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 0, 1]
v_pk_fma_f16 v89, v89, -2.0, 1.0 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 0, 0, 1]
v_mov_b32 v91, 1002585026
v_cmp_gt_f16 [vcc_lo, vcc_hi], abs(v66), v91
v_cndmask_b32 v86, v86, v85, [vcc_lo, vcc_hi] dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 src1_sel:WORD_0
v_mov_b32 v91, 1002585026
v_cmp_gt_f16 [vcc_lo, vcc_hi], abs(v70), v91
v_cndmask_b32 v90, v90, v89, [vcc_lo, vcc_hi] dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 src1_sel:WORD_0
v_mov_b32 v91, 1002585026
v_cmp_gt_f16 [vcc_lo, vcc_hi], abs(v66), v91 src0_sel:WORD_1 src1_sel:WORD_1
v_cndmask_b32 v86, v86, v85, [vcc_lo, vcc_hi] dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_1
v_mov_b32 v91, 1002585026
v_cmp_gt_f16 [vcc_lo, vcc_hi], abs(v70), v91 src0_sel:WORD_1 src1_sel:WORD_1
v_cndmask_b32 v90, v90, v89, [vcc_lo, vcc_hi] dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_1
v_mov_b32 v85, 2147450879
v_bfi_b32 v66, v85, v86, v66
v_mov_b32 v89, 2147450879
v_bfi_b32 v70, v89, v90, v70
v_pk_mul_f16 v66, v66, s46 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f16 v70, v70, s46 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
s_setprio 0
buffer_store_short v64, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short v68, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short v66, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short v70, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v64, v78, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v68, v82, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v66, v79, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v70, v83, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s59
s_addc_u32 s69, s69, 0
s_sub_u32 s63, s63, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s78, label_114 - label_1
s_bitcmp1_b32 s53, 0
s_cbranch_scc1 label_0
s_sub_u32 s63, s67, 1
s_cselect_b32 s70, 0, 2147483648
s_mov_b64 [s68, s69], [s34, s35]
s_setpc_b64 [s76, s77]
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
