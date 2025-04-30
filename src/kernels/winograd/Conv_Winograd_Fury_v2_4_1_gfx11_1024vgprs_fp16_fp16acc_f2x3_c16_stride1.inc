/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

.macro _sop1_lit op:req, sdst:req, lit:req
    .long (0b101111101 << 23) | (\sdst << 16) | (\op << 8) | 255
    .long \lit
.endm

.macro _s_mov_b32__sop1_lit sdst:req, lit:req
    _sop1_lit 0, \sdst, \lit
.endm

.macro _vop1 op:req, vdst:req, src:req
    .long (0b0111111 << 25) | (\vdst << 17) | (\op << 9) | \src
.endm

.macro _v_cvt_f16_i16__vop1 vdst:req, vsrc:req
    _vop1 81, \vdst, (\vsrc + /*VGPR*/ 256)
.endm

.macro _v_rcp_f16__vop1 vdst:req, vsrc:req
    _vop1 84, \vdst, (\vsrc + /*VGPR*/ 256)
.endm

.macro _v_exp_f16__vop1 vdst:req, vsrc:req
    _vop1 88, \vdst, (\vsrc + /*VGPR*/ 256)
.endm

.macro _vop3 op:req, vdst:req, src0:req, src1:req, src2:req, opsel:req, abs:req, neg:req
    .long (0b110101 << 26) | (\op << 16) | (\opsel << 11) | (\abs << 8) | \vdst
    .long (\neg << 29) | (\src2 << 18) | (\src1 << 9) | \src0
.endm

.macro _vop3_lit op:req, vdst:req, src0:req, src1:req, src2:req, lit:req, opsel:req, abs:req, neg:req
    .long (0b110101 << 26) | (\op << 16) | (\opsel << 11) | (\abs << 8) | \vdst
    .long (\neg << 29) | (\src2 << 18) | (\src1 << 9) | \src0
    .long \lit
.endm

.macro _v_cvt_f16_i16__vop3 vdst:req, vsrc:req, opsel:req
    _vop3 465, \vdst, (\vsrc + /*VGPR*/ 256), 0, 0, \opsel, 0, 0
.endm

.macro _v_rcp_f16__vop3 vdst:req, vsrc:req, opsel:req
    _vop3 468, \vdst, (\vsrc + /*VGPR*/ 256), 0, 0, \opsel, 0, 0
.endm

.macro _v_exp_f16__vop3 vdst:req, vsrc:req, opsel:req
    _vop3 472, \vdst, (\vsrc + /*VGPR*/ 256), 0, 0, \opsel, 0, 0
.endm

.macro _v_cndmask_b16__vop3 vdst:req, vsrc0:req, vsrc1:req, src2:req, opsel:req
    _vop3 605, \vdst, (\vsrc0 + /*VGPR*/ 256), (\vsrc1 + /*VGPR*/ 256), \src2, \opsel, 0, 0
.endm

.macro _v_cmp_gt_f16__vop3_s_lit sdst:req, ssrc0:req, lit:req, opsel:req, abs:req
    _vop3_lit 4, \sdst, \ssrc0, 255, 0, \lit, \opsel, \abs, 0
.endm

.macro _v_cmp_gt_f16__vop3_v_lit sdst:req, vsrc0:req, lit:req, opsel:req, abs:req
    _vop3_lit 4, \sdst, (\vsrc0 + /*VGPR*/ 256), 255, 0, \lit, \opsel, \abs, 0
.endm

.macro _v_cmp_lt_u16__vop3 sdst:req, vsrc0:req, ssrc1:req, opsel:req
    _vop3 57, \sdst, (\vsrc0 + /*VGPR*/ 256), \ssrc1, 0, \opsel, 0, 0
.endm

.macro _v_cmpx_lt_u32__vop3 sdst:req, vsrc0:req, ssrc1:req
    _vop3 201, \sdst, (\vsrc0 + /*VGPR*/ 256), \ssrc1, 0, 0, 0, 0
.endm

.macro _vop3p op:req, vdst:req, src0:req, src1:req, src2:req, opsel:req, opsel_hi:req, opsel_hi2:req, neg:req, neg_hi:req
    .long (0b11001100 << 24) | (\op << 16) | (\opsel_hi2 << 14) | (\opsel << 11) | (\neg_hi << 8) | \vdst
    .long (\neg << 29) | (\opsel_hi << 27) | (\src2 << 18) | (\src1 << 9) | \src0
.endm

.macro _vop3p_lit op:req, vdst:req, src0:req, src1:req, src2:req, lit:req, opsel:req, opsel_hi:req, opsel_hi2:req, neg:req, neg_hi:req
    .long (0b11001100 << 24) | (\op << 16) | (\opsel_hi2 << 14) | (\opsel << 11) | (\neg_hi << 8) | \vdst
    .long (\neg << 29) | (\opsel_hi << 27) | (\src2 << 18) | (\src1 << 9) | \src0
    .long \lit
.endm

.macro _v_pk_ashrrev_i16__vop3p vdst:req, src0:req, src1:req, opsel:req, opsel_hi:req, neg:req, neg_hi:req
    _vop3p 6, \vdst, \src0, \src1, 0, \opsel, \opsel_hi, 0, \neg, \neg_hi
.endm

.macro _v_pk_add_u16__vop3p vdst:req, src0:req, src1:req, opsel:req, opsel_hi:req, neg:req, neg_hi:req
    _vop3p 10, \vdst, \src0, \src1, 0, \opsel, \opsel_hi, 0, \neg, \neg_hi
.endm

.macro _v_pk_sub_u16__vop3p vdst:req, src0:req, src1:req, opsel:req, opsel_hi:req, neg:req, neg_hi:req
    _vop3p 11, \vdst, \src0, \src1, 0, \opsel, \opsel_hi, 0, \neg, \neg_hi
.endm

.macro _v_pk_min_u16__vop3p vdst:req, src0:req, src1:req, opsel:req, opsel_hi:req, neg:req, neg_hi:req
    _vop3p 13, \vdst, \src0, \src1, 0, \opsel, \opsel_hi, 0, \neg, \neg_hi
.endm

.macro _v_pk_add_f16__vop3p vdst:req, src0:req, src1:req, opsel:req, opsel_hi:req, neg:req, neg_hi:req
    _vop3p 15, \vdst, \src0, \src1, 0, \opsel, \opsel_hi, 0, \neg, \neg_hi
.endm

.macro _v_pk_add_f16__vop3p_lit vdst:req, lit:req, src1:req, opsel:req, opsel_hi:req
    _vop3p_lit 15, \vdst, 255, \src1, 0, \lit, \opsel, \opsel_hi, 0, 0, 0
.endm

.macro _v_pk_mul_f16__vop3p vdst:req, src0:req, src1:req, opsel:req, opsel_hi:req, neg:req, neg_hi:req
    _vop3p 16, \vdst, \src0, \src1, 0, \opsel, \opsel_hi, 0, \neg, \neg_hi
.endm

.macro _v_pk_mul_f16__vop3p_lit vdst:req, lit:req, src1:req, opsel:req, opsel_hi:req
    _vop3p_lit 16, \vdst, 255, \src1, 0, \lit, \opsel, \opsel_hi, 0, 0, 0
.endm

.macro _v_pk_min_f16__vop3p vdst:req, src0:req, src1:req, opsel:req, opsel_hi:req, neg:req, neg_hi:req
    _vop3p 17, \vdst, \src0, \src1, 0, \opsel, \opsel_hi, 0, \neg, \neg_hi
.endm

.macro _v_pk_max_f16__vop3p vdst:req, src0:req, src1:req, opsel:req, opsel_hi:req, neg:req, neg_hi:req
    _vop3p 18, \vdst, \src0, \src1, 0, \opsel, \opsel_hi, 0, \neg, \neg_hi
.endm

s_version 0x2006
s_set_inst_prefetch_distance 0x3
s_mov_b32 s0, 0
v_lshlrev_b32 v1, 7, v0
s_getpc_b64 s[8:9]
s_mov_b32 s10, 0x5ccc
s_mov_b32 s11, 0x31014000
buffer_load_b32 v2, v1, s[8:11], 0 offen
s_waitcnt vmcnt(0)
s_getpc_b64 s[6:7]
s_load_b512 s[8:23], s[2:3], null
s_load_b512 s[24:39], s[2:3], 0x40
s_load_b512 s[40:55], s[2:3], 0x80
s_load_b256 s[56:63], s[2:3], 0xc0
s_load_b64 s[64:65], s[2:3], 0xe0
v_and_b32 v8, 0xff, v0
v_lshrrev_b32 v9, 1, v8
v_and_b32 v10, 1, v0
v_add_nc_u32 v5, v9, 32
v_bfi_b32 v6, 31, v8, v9
v_bfe_u32 v4, v8, 5, 1
v_bfi_b32 v6, 0xbf, v6, v5
v_and_b32 v2, 31, v8
v_lshrrev_b32 v6, 5, v6
v_lshrrev_b32 v7, 6, v8
v_lshlrev_b32 v2, 4, v2
v_and_b32 v3, 31, v9
v_mad_u32_u24 v2, v4, 0x900, v2
v_lshlrev_b32 v3, 4, v3
v_xor_b32 v5, 3, v6
v_mad_u32_u16 v3, 0x480, v7, v3 op_sel:[0,0,0,0]
v_mad_u32_u24 v1, v5, 0x240, v2
v_mad_u32_u16 v3, 0x1240, v10, v3 op_sel:[0,0,0,0]
v_mad_u32_u24 v2, v6, 0x240, v2
s_waitcnt expcnt(0) lgkmcnt(0) vmcnt(0)
s_bitcmp1_b32 s14, 6
s_cbranch_scc0 14
s_load_b64 s[16:17], s[16:17], null
s_load_b64 s[20:21], s[20:21], null
s_load_b64 s[18:19], s[18:19], null
s_cmp_eq_u64 0, s[60:61]
s_cbranch_scc1 2
s_load_b64 s[60:61], s[60:61], null
s_cmp_eq_u64 0, s[30:31]
s_cbranch_scc1 2
s_load_b64 s[30:31], s[30:31], null
s_bitcmp1_b32 s14, 3
s_cbranch_scc0 2
s_setreg_imm32_b32 hwreg(HW_REG_MODE, 0, 8), 0xf0
s_cmp_eq_u32 s13, 0x60
s_cbranch_scc0 16
s_mul_i32 s1, s4, 0xab
s_lshr_b32 s1, s1, 10
s_mul_i32 s23, s1, 6
s_sub_u32 s23, s4, s23
s_bfe_u32 s15, s1, 0x20000
s_bfe_u32 s22, s1, 0x10002
s_bfe_u32 s5, s1, 0x10003
s_mov_b32 s45, s23
s_lshl1_add_u32 s45, s45, s22
s_lshl2_add_u32 s45, s45, s15
s_lshl1_add_u32 s45, s45, s5
s_mov_b32 s4, s45
s_waitcnt expcnt(0) lgkmcnt(0) vmcnt(0)
s_bitcmp1_b32 s14, 13
s_cbranch_scc0 10
s_add_u32 s16, s16, s34
s_addc_u32 s17, s17, s35
s_add_u32 s20, s20, s38
s_addc_u32 s21, s21, s39
s_add_u32 s18, s18, s36
s_addc_u32 s19, s19, s37
s_cmp_eq_u64 0, s[30:31]
s_cselect_b64 s[40:41], 0, s[40:41]
s_add_u32 s30, s30, s40
s_addc_u32 s31, s31, s41
s_add_u32 s15, s12, 15
s_lshr_b32 s15, s15, 4
v_cvt_f32_u32 v4, s15
v_rcp_f32 v4, v4
v_mul_f32 v4, 0x47800000, v4
v_cvt_floor_i32_f32 v4, v4
v_mad_u32_u24 v5, v4, s13, s13
v_lshrrev_b32 v5, 16, v5
v_cvt_f32_u32 v4, v5
v_rcp_f32 v4, v4
v_mul_f32 v4, 0x47800000, v4
v_cvt_floor_i32_f32 v4, v4
v_mad_u32_u24 v6, v4, s4, s4
v_lshrrev_b32 v6, 16, v6
v_readfirstlane_b32 s1, v5
v_readfirstlane_b32 s22, v6
s_mul_i32 s5, s22, s1
s_sub_u32 s5, s4, s5
s_cmp_ge_u32 s22, s15
s_cbranch_scc1 5809
s_mul_i32 s13, s1, s15
s_mul_i32 s23, s22, 16
s_sub_u32 s12, s12, s23
s_min_u32 s12, s12, 16
s_mul_i32 s34, s23, s46
s_mul_hi_u32 s35, s23, s46
s_lshl_b64 s[34:35], s[34:35], 1
s_add_u32 s18, s34, s18
s_addc_u32 s19, s35, s19
s_lshr_b32 s35, s23, 0
s_mul_i32 s34, s35, s51
s_mul_hi_u32 s35, s35, s51
s_lshl_b64 s[34:35], s[34:35], 1
s_add_u32 s20, s34, s20
s_addc_u32 s21, s35, s21
s_lshl_b32 s34, s23, 1
s_cmp_eq_u64 s[30:31], 0
s_cselect_b32 s34, 0, s34
s_add_u32 s30, s30, s34
s_addc_u32 s31, s31, 0
v_cmp_lt_u32 vcc, v0, 0x100
s_cbranch_vccz 2749
v_and_b32 v20, 0xff, v0
v_lshrrev_b32 v21, 1, v20
v_bfe_u32 v17, v20, 3, 1
v_bfe_u32 v16, v20, 2, 1
v_mad_u32_u16 v17, v17, 16, 0 op_sel:[0,0,0,0]
v_mad_u32_u16 v14, v16, 0x1240, v17 op_sel:[0,0,0,0]
v_bfe_u32 v16, v20, 0, 2
v_mad_u32_u16 v14, v16, 0x90, v14 op_sel:[0,0,0,0]
v_bfe_u32 v17, v20, 4, 2
v_mad_u32_u16 v14, v17, 32, v14 op_sel:[0,0,0,0]
v_bfe_u32 v16, v20, 6, 1
v_mad_u32_u16 v14, v16, 0x480, v14 op_sel:[0,0,0,0]
v_bfe_u32 v16, v20, 7, 1
v_mad_u32_u16 v14, v16, 0x900, v14 op_sel:[0,0,0,0]
v_bfe_u32 v18, v20, 1, 2
v_mad_u32_u16 v13, v18, 32, 0 op_sel:[0,0,0,0]
v_bfe_u32 v19, v20, 3, 1
v_mad_u32_u16 v13, v19, 0x480, v13 op_sel:[0,0,0,0]
v_add_nc_u32 v18, v21, 32
v_bfi_b32 v18, 0xbf, v20, v18
v_bfe_u32 v18, v18, 6, 2
v_mad_u32_u16 v13, v18, 0x90, v13 op_sel:[0,0,0,0]
v_xor_b32 v16, v0, v0 quad_perm:[2,3,2,1]
v_xor_b32 v17, v0, v0 quad_perm:[0,0,3,3]
v_sub_nc_u16 v16, v16, v17 op_sel:[0,0,0]
v_cvt_f16_i16 v15, v16
_v_cvt_f16_i16__vop1 (15 | /*op_sel*/ 0x80), 17
_v_pk_mul_f16__vop3p 15, 271, 240, 0x0, 0x1, 0x0, 0x0
v_bfe_u32 v16, v0, 6, 1
v_and_b32 v5, 63, v0
v_cmp_eq_u32 vcc, v16, 1
v_cndmask_b32 v16, 0, 0x400, vcc
v_cndmask_b32 v17, 0, 0x100, vcc
v_lshl_add_u32 v6, v5, 2, 0
v_lshl_add_u32 v5, v5, 4, v16
s_mov_b32 s23, 4
s_mov_b32 s34, 0
s_mov_b32 s40, 0xbc00c000
v_readfirstlane_b32 s74, v0
s_and_b32 null, 64, s74
s_cmov_b32 s40, 0x3c00c000
s_lshl_b32 s49, s43, 1
s_lshl_b32 s53, s47, 1
s_lshl_b32 s75, s49, 3
s_lshl_b32 s76, s53, 3
s_and_b32 null, 0x80, s74
s_cselect_b32 s75, s75, 0
s_cselect_b32 s76, s76, 0
s_cselect_b32 s22, 8, 0
s_sub_u32 s22, s9, s22
s_cmov_b32 s22, 0
s_mov_b32 s35, 0x11014000
s_bitcmp1_b32 s14, 4
s_cselect_b32 s77, 0, 0x8000000
s_and_b32 s35, 0xf7ffffff, s35
s_or_b32 s35, s35, s77
s_and_b32 s17, s17, 0xffff
s_add_u32 s17, s17, 0x20000
s_and_b32 s19, s19, 0xffff
s_add_u32 s19, s19, 0x20000
s_add_u32 s16, s16, s75
s_addc_u32 s17, s17, 0
s_add_u32 s18, s18, s76
s_addc_u32 s19, s19, 0
s_mov_b64 s[36:37], s[16:17]
s_mov_b32 s38, 0x80000000
s_mov_b32 s39, 0
s_getpc_b64 s[64:65]
v_cmp_lt_u32 vcc, v0, 0x80
s_cmp_gt_u32 vcc_lo, 0
s_mov_b32 s74, 0x23d8
s_mov_b32 s76, 0x1a58
s_cmov_b32 s74, 0x1e98
s_cmov_b32 s76, 0x1618
s_mov_b32 s75, 0x2654
s_mov_b32 s77, 0x1c54
s_cmov_b32 s75, 0x2114
s_cmov_b32 s77, 0x1814
s_add_u32 s66, s64, s74
s_addc_u32 s67, s65, 0
s_add_u32 s70, s64, s76
s_addc_u32 s71, s65, 0
s_add_u32 s68, s64, s75
s_addc_u32 s69, s65, 0
s_add_u32 s72, s64, s77
s_addc_u32 s73, s65, 0
s_mov_b32 s45, 0
v_mov_b32 v4, 0
s_mov_b32 s56, 0x190
s_bitcmp1_b32 s45, 3
s_cselect_b64 s[64:65], s[66:67], s[70:71]
s_bitcmp1_b32 s45, 2
s_cselect_b32 s56, s56, 0x2b8
s_setprio 2
s_waitcnt vmcnt(32)
_v_pk_add_f16__vop3p 160, 272, 273, 0x0, 0x3, 0x1, 0x1
_v_pk_add_f16__vop3p 161, 308, 341, 0x0, 0x3, 0x1, 0x1
_v_pk_add_f16__vop3p 162, 360, 377, 0x0, 0x3, 0x1, 0x1
_v_pk_add_f16__vop3p 163, 396, 397, 0x0, 0x3, 0x1, 0x1
v_pk_fma_f16 v164, v16, s40, v34 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v165, v52, s40, v86 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v166, v104, s40, v122 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v167, v140, s40, v142 op_sel:[0,1,0] op_sel_hi:[1,1,1]
buffer_load_d16_b16 v17, v7, s[36:39], 0 idxen
buffer_load_d16_b16 v16, v9, s[36:39], 0 idxen
buffer_load_d16_b16 v121, v8, s[36:39], 0 idxen
buffer_load_d16_b16 v104, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_hi_b16 v17, v7, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v16, v9, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v121, v8, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v104, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_b16 v85, v7, s[36:39], 0 idxen
buffer_load_d16_b16 v52, v9, s[36:39], 0 idxen
buffer_load_d16_b16 v141, v8, s[36:39], 0 idxen
buffer_load_d16_b16 v140, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_hi_b16 v85, v7, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v52, v9, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v141, v8, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v140, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
s_swappc_b64 s[64:65], s[64:65]
s_branch 5561
_s_mov_b32__sop1_lit 56, 0x4
s_bitcmp1_b32 s45, 3
s_cselect_b64 s[64:65], s[66:67], s[70:71]
s_bitcmp1_b32 s45, 2
s_cselect_b32 s56, s56, 0x12c
s_setprio 2
s_waitcnt vmcnt(32)
_v_pk_mul_f16__vop3p 160, 273, 271, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 161, 341, 271, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 162, 377, 271, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 163, 397, 271, 0x0, 0x1, 0x0, 0x0
v_mov_b32 v17, v160 quad_perm:[1,0,3,2]
v_mov_b32 v85, v161 quad_perm:[1,0,3,2]
v_mov_b32 v121, v162 quad_perm:[1,0,3,2]
v_mov_b32 v141, v163 quad_perm:[1,0,3,2]
v_pk_fma_f16 v160, v17, v15, v160 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v161, v85, v15, v161 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v162, v121, v15, v162 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v163, v141, v15, v163 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_mov_b32 v17, v160 quad_perm:[2,3,0,1]
v_mov_b32 v85, v161 quad_perm:[2,3,0,1]
v_mov_b32 v121, v162 quad_perm:[2,3,0,1]
v_mov_b32 v141, v163 quad_perm:[2,3,0,1]
v_pk_fma_f16 v160, v17, v15, v160 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v161, v85, v15, v161 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v162, v121, v15, v162 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v163, v141, v15, v163 op_sel:[0,1,0] op_sel_hi:[1,1,1]
buffer_load_d16_b16 v17, v7, s[36:39], 0 idxen
buffer_load_d16_b16 v16, v9, s[36:39], 0 idxen
buffer_load_d16_b16 v121, v8, s[36:39], 0 idxen
buffer_load_d16_b16 v104, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_hi_b16 v17, v7, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v16, v9, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v121, v8, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v104, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_b16 v85, v7, s[36:39], 0 idxen
buffer_load_d16_b16 v52, v9, s[36:39], 0 idxen
buffer_load_d16_b16 v141, v8, s[36:39], 0 idxen
buffer_load_d16_b16 v140, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_hi_b16 v85, v7, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v52, v9, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v141, v8, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v140, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
s_swappc_b64 s[64:65], s[64:65]
s_branch 5462
s_mov_b32 s56, 0x18c
s_bitcmp1_b32 s45, 4
s_cselect_b64 s[64:65], s[68:69], s[72:73]
s_bitcmp1_b32 s45, 2
s_cselect_b32 s56, s56, 0x2b8
s_setprio 2
v_pk_fma_f16 v160, v34, s40, v164 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v161, v86, s40, v165 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v162, v122, s40, v166 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v163, v142, s40, v167 op_sel:[0,0,0] op_sel_hi:[1,0,1]
_v_pk_add_f16__vop3p 164, 290, 291, 0x0, 0x3, 0x1, 0x1
_v_pk_add_f16__vop3p 165, 342, 343, 0x0, 0x3, 0x1, 0x1
_v_pk_add_f16__vop3p 166, 378, 379, 0x0, 0x3, 0x1, 0x1
_v_pk_add_f16__vop3p 167, 398, 399, 0x0, 0x3, 0x1, 0x1
buffer_load_d16_b16 v34, v7, s[36:39], 0 idxen
buffer_load_d16_b16 v35, v9, s[36:39], 0 idxen
buffer_load_d16_b16 v122, v8, s[36:39], 0 idxen
buffer_load_d16_b16 v123, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_hi_b16 v34, v7, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v35, v9, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v122, v8, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v123, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_b16 v86, v7, s[36:39], 0 idxen
buffer_load_d16_b16 v87, v9, s[36:39], 0 idxen
buffer_load_d16_b16 v142, v8, s[36:39], 0 idxen
buffer_load_d16_b16 v143, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_hi_b16 v86, v7, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v87, v9, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v142, v8, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v143, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
s_swappc_b64 s[64:65], s[64:65]
s_branch 5388
_s_mov_b32__sop1_lit 56, 0x4
s_bitcmp1_b32 s45, 4
s_cselect_b64 s[64:65], s[68:69], s[72:73]
s_bitcmp1_b32 s45, 2
s_cselect_b32 s56, s56, 0x130
s_setprio 2
_v_pk_mul_f16__vop3p 160, 290, 271, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 161, 342, 271, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 162, 378, 271, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 163, 398, 271, 0x0, 0x1, 0x0, 0x0
v_mov_b32 v34, v160 quad_perm:[1,0,3,2]
v_mov_b32 v86, v161 quad_perm:[1,0,3,2]
v_mov_b32 v122, v162 quad_perm:[1,0,3,2]
v_mov_b32 v142, v163 quad_perm:[1,0,3,2]
v_pk_fma_f16 v160, v34, v15, v160 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v161, v86, v15, v161 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v162, v122, v15, v162 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v163, v142, v15, v163 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_mov_b32 v34, v160 quad_perm:[2,3,0,1]
v_mov_b32 v86, v161 quad_perm:[2,3,0,1]
v_mov_b32 v122, v162 quad_perm:[2,3,0,1]
v_mov_b32 v142, v163 quad_perm:[2,3,0,1]
v_pk_fma_f16 v160, v34, v15, v160 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v161, v86, v15, v161 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v162, v122, v15, v162 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v163, v142, v15, v163 op_sel:[0,1,0] op_sel_hi:[1,1,1]
buffer_load_d16_b16 v34, v7, s[36:39], 0 idxen
buffer_load_d16_b16 v35, v9, s[36:39], 0 idxen
buffer_load_d16_b16 v122, v8, s[36:39], 0 idxen
buffer_load_d16_b16 v123, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_hi_b16 v34, v7, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v35, v9, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v122, v8, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v123, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_b16 v86, v7, s[36:39], 0 idxen
buffer_load_d16_b16 v87, v9, s[36:39], 0 idxen
buffer_load_d16_b16 v142, v8, s[36:39], 0 idxen
buffer_load_d16_b16 v143, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_hi_b16 v86, v7, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v87, v9, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v142, v8, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v143, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
s_swappc_b64 s[64:65], s[64:65]
s_branch 5290
s_mov_b32 s56, 0x190
s_bitcmp1_b32 s45, 3
s_cselect_b64 s[64:65], s[66:67], s[70:71]
s_bitcmp1_b32 s45, 2
s_cselect_b32 s56, s56, 0x2b8
s_setprio 2
s_waitcnt vmcnt(32)
_v_pk_add_f16__vop3p 160, 403, 402, 0x0, 0x3, 0x1, 0x1
_v_pk_add_f16__vop3p 161, 407, 406, 0x0, 0x3, 0x1, 0x1
_v_pk_add_f16__vop3p 162, 411, 410, 0x0, 0x3, 0x1, 0x1
_v_pk_add_f16__vop3p 163, 415, 414, 0x0, 0x3, 0x1, 0x1
v_pk_fma_f16 v164, v147, s40, v144 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v165, v151, s40, v148 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v166, v155, s40, v152 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v167, v159, s40, v156 op_sel:[0,1,0] op_sel_hi:[1,1,1]
buffer_load_d16_b16 v146, v7, s[36:39], 0 idxen
buffer_load_d16_b16 v147, v9, s[36:39], 0 idxen
buffer_load_d16_b16 v154, v8, s[36:39], 0 idxen
buffer_load_d16_b16 v155, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_hi_b16 v146, v7, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v147, v9, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v154, v8, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v155, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_b16 v150, v7, s[36:39], 0 idxen
buffer_load_d16_b16 v151, v9, s[36:39], 0 idxen
buffer_load_d16_b16 v158, v8, s[36:39], 0 idxen
buffer_load_d16_b16 v159, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_hi_b16 v150, v7, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v151, v9, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v158, v8, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v159, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
s_swappc_b64 s[64:65], s[64:65]
s_branch 5215
_s_mov_b32__sop1_lit 56, 0x4
s_bitcmp1_b32 s45, 3
s_cselect_b64 s[64:65], s[66:67], s[70:71]
s_bitcmp1_b32 s45, 2
s_cselect_b32 s56, s56, 0x12c
s_setprio 2
s_waitcnt vmcnt(32)
_v_pk_mul_f16__vop3p 160, 402, 271, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 161, 406, 271, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 162, 410, 271, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 163, 414, 271, 0x0, 0x1, 0x0, 0x0
v_mov_b32 v146, v160 quad_perm:[1,0,3,2]
v_mov_b32 v150, v161 quad_perm:[1,0,3,2]
v_mov_b32 v154, v162 quad_perm:[1,0,3,2]
v_mov_b32 v158, v163 quad_perm:[1,0,3,2]
v_pk_fma_f16 v160, v146, v15, v160 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v161, v150, v15, v161 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v162, v154, v15, v162 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v163, v158, v15, v163 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_mov_b32 v146, v160 quad_perm:[2,3,0,1]
v_mov_b32 v150, v161 quad_perm:[2,3,0,1]
v_mov_b32 v154, v162 quad_perm:[2,3,0,1]
v_mov_b32 v158, v163 quad_perm:[2,3,0,1]
v_pk_fma_f16 v160, v146, v15, v160 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v161, v150, v15, v161 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v162, v154, v15, v162 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v163, v158, v15, v163 op_sel:[0,1,0] op_sel_hi:[1,1,1]
buffer_load_d16_b16 v146, v7, s[36:39], 0 idxen
buffer_load_d16_b16 v147, v9, s[36:39], 0 idxen
buffer_load_d16_b16 v154, v8, s[36:39], 0 idxen
buffer_load_d16_b16 v155, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_hi_b16 v146, v7, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v147, v9, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v154, v8, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v155, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_b16 v150, v7, s[36:39], 0 idxen
buffer_load_d16_b16 v151, v9, s[36:39], 0 idxen
buffer_load_d16_b16 v158, v8, s[36:39], 0 idxen
buffer_load_d16_b16 v159, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_hi_b16 v150, v7, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v151, v9, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v158, v8, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v159, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
s_swappc_b64 s[64:65], s[64:65]
s_branch 5116
s_mov_b32 s56, 0x18c
s_bitcmp1_b32 s45, 4
s_cselect_b64 s[64:65], s[68:69], s[72:73]
s_bitcmp1_b32 s45, 2
s_cselect_b32 s56, s56, 0x2b8
s_setprio 2
v_pk_fma_f16 v160, v144, s40, v164 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v161, v148, s40, v165 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v162, v152, s40, v166 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v163, v156, s40, v167 op_sel:[0,0,0] op_sel_hi:[1,0,1]
_v_pk_add_f16__vop3p 164, 400, 401, 0x0, 0x3, 0x1, 0x1
_v_pk_add_f16__vop3p 165, 404, 405, 0x0, 0x3, 0x1, 0x1
_v_pk_add_f16__vop3p 166, 408, 409, 0x0, 0x3, 0x1, 0x1
_v_pk_add_f16__vop3p 167, 412, 413, 0x0, 0x3, 0x1, 0x1
buffer_load_d16_b16 v144, v7, s[36:39], 0 idxen
buffer_load_d16_b16 v145, v9, s[36:39], 0 idxen
buffer_load_d16_b16 v152, v8, s[36:39], 0 idxen
buffer_load_d16_b16 v153, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_hi_b16 v144, v7, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v145, v9, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v152, v8, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v153, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_b16 v148, v7, s[36:39], 0 idxen
buffer_load_d16_b16 v149, v9, s[36:39], 0 idxen
buffer_load_d16_b16 v156, v8, s[36:39], 0 idxen
buffer_load_d16_b16 v157, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_hi_b16 v148, v7, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v149, v9, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v156, v8, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v157, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
s_swappc_b64 s[64:65], s[64:65]
s_branch 5042
_s_mov_b32__sop1_lit 56, 0x4
s_bitcmp1_b32 s45, 4
s_cselect_b64 s[64:65], s[68:69], s[72:73]
s_bitcmp1_b32 s45, 2
s_cselect_b32 s56, s56, 0x130
s_setprio 2
_v_pk_mul_f16__vop3p 160, 400, 271, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 161, 404, 271, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 162, 408, 271, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 163, 412, 271, 0x0, 0x1, 0x0, 0x0
v_mov_b32 v144, v160 quad_perm:[1,0,3,2]
v_mov_b32 v148, v161 quad_perm:[1,0,3,2]
v_mov_b32 v152, v162 quad_perm:[1,0,3,2]
v_mov_b32 v156, v163 quad_perm:[1,0,3,2]
v_pk_fma_f16 v160, v144, v15, v160 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v161, v148, v15, v161 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v162, v152, v15, v162 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v163, v156, v15, v163 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_mov_b32 v144, v160 quad_perm:[2,3,0,1]
v_mov_b32 v148, v161 quad_perm:[2,3,0,1]
v_mov_b32 v152, v162 quad_perm:[2,3,0,1]
v_mov_b32 v156, v163 quad_perm:[2,3,0,1]
v_pk_fma_f16 v160, v144, v15, v160 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v161, v148, v15, v161 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v162, v152, v15, v162 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v163, v156, v15, v163 op_sel:[0,1,0] op_sel_hi:[1,1,1]
buffer_load_d16_b16 v144, v7, s[36:39], 0 idxen
buffer_load_d16_b16 v145, v9, s[36:39], 0 idxen
buffer_load_d16_b16 v152, v8, s[36:39], 0 idxen
buffer_load_d16_b16 v153, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_hi_b16 v144, v7, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v145, v9, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v152, v8, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v153, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_b16 v148, v7, s[36:39], 0 idxen
buffer_load_d16_b16 v149, v9, s[36:39], 0 idxen
buffer_load_d16_b16 v156, v8, s[36:39], 0 idxen
buffer_load_d16_b16 v157, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_hi_b16 v148, v7, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v149, v9, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v156, v8, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v157, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
s_swappc_b64 s[64:65], s[64:65]
s_branch 4944
s_mov_b32 s56, 0x190
s_bitcmp1_b32 s45, 3
s_cselect_b64 s[64:65], s[66:67], s[70:71]
s_bitcmp1_b32 s45, 2
s_cselect_b32 s56, s56, 0x2b8
s_setprio 2
s_waitcnt vmcnt(32)
_v_pk_add_f16__vop3p 160, 272, 273, 0x0, 0x3, 0x1, 0x1
_v_pk_add_f16__vop3p 161, 308, 341, 0x0, 0x3, 0x1, 0x1
_v_pk_add_f16__vop3p 162, 291, 290, 0x0, 0x3, 0x1, 0x1
_v_pk_add_f16__vop3p 163, 343, 342, 0x0, 0x3, 0x1, 0x1
v_pk_fma_f16 v164, v16, s40, v121 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v165, v52, s40, v141 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v166, v35, s40, v122 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v167, v87, s40, v142 op_sel:[0,1,0] op_sel_hi:[1,1,1]
buffer_load_d16_b16 v17, v7, s[36:39], 0 idxen
buffer_load_d16_b16 v16, v9, s[36:39], 0 idxen
buffer_load_d16_b16 v34, v8, s[36:39], 0 idxen
buffer_load_d16_b16 v35, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_hi_b16 v17, v7, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v16, v9, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v34, v8, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v35, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_b16 v85, v7, s[36:39], 0 idxen
buffer_load_d16_b16 v52, v9, s[36:39], 0 idxen
buffer_load_d16_b16 v86, v8, s[36:39], 0 idxen
buffer_load_d16_b16 v87, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_hi_b16 v85, v7, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v52, v9, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v86, v8, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v87, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
s_swappc_b64 s[64:65], s[64:65]
s_branch 4869
_s_mov_b32__sop1_lit 56, 0x4
s_bitcmp1_b32 s45, 3
s_cselect_b64 s[64:65], s[66:67], s[70:71]
s_bitcmp1_b32 s45, 2
s_cselect_b32 s56, s56, 0x12c
s_setprio 2
s_waitcnt vmcnt(32)
_v_pk_mul_f16__vop3p 160, 273, 271, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 161, 341, 271, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 162, 290, 271, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 163, 342, 271, 0x0, 0x1, 0x0, 0x0
v_mov_b32 v17, v160 quad_perm:[1,0,3,2]
v_mov_b32 v85, v161 quad_perm:[1,0,3,2]
v_mov_b32 v34, v162 quad_perm:[1,0,3,2]
v_mov_b32 v86, v163 quad_perm:[1,0,3,2]
v_pk_fma_f16 v160, v17, v15, v160 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v161, v85, v15, v161 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v162, v34, v15, v162 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v163, v86, v15, v163 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_mov_b32 v17, v160 quad_perm:[2,3,0,1]
v_mov_b32 v85, v161 quad_perm:[2,3,0,1]
v_mov_b32 v34, v162 quad_perm:[2,3,0,1]
v_mov_b32 v86, v163 quad_perm:[2,3,0,1]
v_pk_fma_f16 v160, v17, v15, v160 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v161, v85, v15, v161 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v162, v34, v15, v162 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v163, v86, v15, v163 op_sel:[0,1,0] op_sel_hi:[1,1,1]
buffer_load_d16_b16 v17, v7, s[36:39], 0 idxen
buffer_load_d16_b16 v16, v9, s[36:39], 0 idxen
buffer_load_d16_b16 v34, v8, s[36:39], 0 idxen
buffer_load_d16_b16 v35, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_hi_b16 v17, v7, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v16, v9, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v34, v8, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v35, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_b16 v85, v7, s[36:39], 0 idxen
buffer_load_d16_b16 v52, v9, s[36:39], 0 idxen
buffer_load_d16_b16 v86, v8, s[36:39], 0 idxen
buffer_load_d16_b16 v87, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_hi_b16 v85, v7, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v52, v9, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v86, v8, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v87, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
s_swappc_b64 s[64:65], s[64:65]
s_branch 4770
s_mov_b32 s56, 0x18c
s_bitcmp1_b32 s45, 4
s_cselect_b64 s[64:65], s[68:69], s[72:73]
s_bitcmp1_b32 s45, 2
s_cselect_b32 s56, s56, 0x2b8
s_setprio 2
v_pk_fma_f16 v160, v121, s40, v164 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v161, v141, s40, v165 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v162, v122, s40, v166 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v163, v142, s40, v167 op_sel:[0,0,0] op_sel_hi:[1,0,1]
_v_pk_add_f16__vop3p 164, 377, 360, 0x0, 0x3, 0x1, 0x1
_v_pk_add_f16__vop3p 165, 397, 396, 0x0, 0x3, 0x1, 0x1
_v_pk_add_f16__vop3p 166, 378, 379, 0x0, 0x3, 0x1, 0x1
_v_pk_add_f16__vop3p 167, 398, 399, 0x0, 0x3, 0x1, 0x1
buffer_load_d16_b16 v121, v7, s[36:39], 0 idxen
buffer_load_d16_b16 v104, v9, s[36:39], 0 idxen
buffer_load_d16_b16 v122, v8, s[36:39], 0 idxen
buffer_load_d16_b16 v123, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_hi_b16 v121, v7, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v104, v9, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v122, v8, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v123, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_b16 v141, v7, s[36:39], 0 idxen
buffer_load_d16_b16 v140, v9, s[36:39], 0 idxen
buffer_load_d16_b16 v142, v8, s[36:39], 0 idxen
buffer_load_d16_b16 v143, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_hi_b16 v141, v7, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v140, v9, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v142, v8, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v143, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
s_swappc_b64 s[64:65], s[64:65]
s_branch 4696
_s_mov_b32__sop1_lit 56, 0x4
s_bitcmp1_b32 s45, 4
s_cselect_b64 s[64:65], s[68:69], s[72:73]
s_bitcmp1_b32 s45, 2
s_cselect_b32 s56, s56, 0x130
s_setprio 2
_v_pk_mul_f16__vop3p 160, 377, 271, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 161, 397, 271, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 162, 378, 271, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 163, 398, 271, 0x0, 0x1, 0x0, 0x0
v_mov_b32 v121, v160 quad_perm:[1,0,3,2]
v_mov_b32 v141, v161 quad_perm:[1,0,3,2]
v_mov_b32 v122, v162 quad_perm:[1,0,3,2]
v_mov_b32 v142, v163 quad_perm:[1,0,3,2]
v_pk_fma_f16 v160, v121, v15, v160 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v161, v141, v15, v161 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v162, v122, v15, v162 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v163, v142, v15, v163 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_mov_b32 v121, v160 quad_perm:[2,3,0,1]
v_mov_b32 v141, v161 quad_perm:[2,3,0,1]
v_mov_b32 v122, v162 quad_perm:[2,3,0,1]
v_mov_b32 v142, v163 quad_perm:[2,3,0,1]
v_pk_fma_f16 v160, v121, v15, v160 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v161, v141, v15, v161 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v162, v122, v15, v162 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v163, v142, v15, v163 op_sel:[0,1,0] op_sel_hi:[1,1,1]
buffer_load_d16_b16 v121, v7, s[36:39], 0 idxen
buffer_load_d16_b16 v104, v9, s[36:39], 0 idxen
buffer_load_d16_b16 v122, v8, s[36:39], 0 idxen
buffer_load_d16_b16 v123, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_hi_b16 v121, v7, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v104, v9, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v122, v8, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v123, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_b16 v141, v7, s[36:39], 0 idxen
buffer_load_d16_b16 v140, v9, s[36:39], 0 idxen
buffer_load_d16_b16 v142, v8, s[36:39], 0 idxen
buffer_load_d16_b16 v143, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_hi_b16 v141, v7, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v140, v9, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v142, v8, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v143, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
s_swappc_b64 s[64:65], s[64:65]
s_branch 4598
s_mov_b32 s56, 0x190
s_bitcmp1_b32 s45, 3
s_cselect_b64 s[64:65], s[66:67], s[70:71]
s_bitcmp1_b32 s45, 2
s_cselect_b32 s56, s56, 0x2b8
s_setprio 2
s_waitcnt vmcnt(32)
_v_pk_add_f16__vop3p 160, 403, 402, 0x0, 0x3, 0x1, 0x1
_v_pk_add_f16__vop3p 161, 407, 406, 0x0, 0x3, 0x1, 0x1
_v_pk_add_f16__vop3p 162, 401, 400, 0x0, 0x3, 0x1, 0x1
_v_pk_add_f16__vop3p 163, 405, 404, 0x0, 0x3, 0x1, 0x1
v_pk_fma_f16 v164, v147, s40, v154 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v165, v151, s40, v158 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v166, v145, s40, v152 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v167, v149, s40, v156 op_sel:[0,1,0] op_sel_hi:[1,1,1]
buffer_load_d16_b16 v146, v7, s[36:39], 0 idxen
buffer_load_d16_b16 v147, v9, s[36:39], 0 idxen
buffer_load_d16_b16 v144, v8, s[36:39], 0 idxen
buffer_load_d16_b16 v145, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_hi_b16 v146, v7, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v147, v9, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v144, v8, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v145, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_b16 v150, v7, s[36:39], 0 idxen
buffer_load_d16_b16 v151, v9, s[36:39], 0 idxen
buffer_load_d16_b16 v148, v8, s[36:39], 0 idxen
buffer_load_d16_b16 v149, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_hi_b16 v150, v7, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v151, v9, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v148, v8, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v149, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
s_swappc_b64 s[64:65], s[64:65]
s_branch 4523
_s_mov_b32__sop1_lit 56, 0x4
s_bitcmp1_b32 s45, 3
s_cselect_b64 s[64:65], s[66:67], s[70:71]
s_bitcmp1_b32 s45, 2
s_cselect_b32 s56, s56, 0x12c
s_setprio 2
s_waitcnt vmcnt(32)
_v_pk_mul_f16__vop3p 160, 402, 271, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 161, 406, 271, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 162, 400, 271, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 163, 404, 271, 0x0, 0x1, 0x0, 0x0
v_mov_b32 v146, v160 quad_perm:[1,0,3,2]
v_mov_b32 v150, v161 quad_perm:[1,0,3,2]
v_mov_b32 v144, v162 quad_perm:[1,0,3,2]
v_mov_b32 v148, v163 quad_perm:[1,0,3,2]
v_pk_fma_f16 v160, v146, v15, v160 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v161, v150, v15, v161 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v162, v144, v15, v162 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v163, v148, v15, v163 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_mov_b32 v146, v160 quad_perm:[2,3,0,1]
v_mov_b32 v150, v161 quad_perm:[2,3,0,1]
v_mov_b32 v144, v162 quad_perm:[2,3,0,1]
v_mov_b32 v148, v163 quad_perm:[2,3,0,1]
v_pk_fma_f16 v160, v146, v15, v160 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v161, v150, v15, v161 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v162, v144, v15, v162 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v163, v148, v15, v163 op_sel:[0,1,0] op_sel_hi:[1,1,1]
buffer_load_d16_b16 v146, v7, s[36:39], 0 idxen
buffer_load_d16_b16 v147, v9, s[36:39], 0 idxen
buffer_load_d16_b16 v144, v8, s[36:39], 0 idxen
buffer_load_d16_b16 v145, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_hi_b16 v146, v7, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v147, v9, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v144, v8, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v145, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_b16 v150, v7, s[36:39], 0 idxen
buffer_load_d16_b16 v151, v9, s[36:39], 0 idxen
buffer_load_d16_b16 v148, v8, s[36:39], 0 idxen
buffer_load_d16_b16 v149, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_hi_b16 v150, v7, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v151, v9, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v148, v8, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v149, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
s_swappc_b64 s[64:65], s[64:65]
s_branch 4424
s_mov_b32 s56, 0xffffebec
s_bitcmp1_b32 s45, 4
s_cselect_b64 s[64:65], s[68:69], s[72:73]
s_bitcmp1_b32 s45, 2
s_cselect_b32 s56, s56, 0xffffed18
s_setprio 2
v_pk_fma_f16 v160, v154, s40, v164 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v161, v158, s40, v165 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v162, v152, s40, v166 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v163, v156, s40, v167 op_sel:[0,0,0] op_sel_hi:[1,0,1]
_v_pk_add_f16__vop3p 164, 410, 411, 0x0, 0x3, 0x1, 0x1
_v_pk_add_f16__vop3p 165, 414, 415, 0x0, 0x3, 0x1, 0x1
_v_pk_add_f16__vop3p 166, 408, 409, 0x0, 0x3, 0x1, 0x1
_v_pk_add_f16__vop3p 167, 412, 413, 0x0, 0x3, 0x1, 0x1
buffer_load_d16_b16 v154, v7, s[36:39], 0 idxen
buffer_load_d16_b16 v155, v9, s[36:39], 0 idxen
buffer_load_d16_b16 v152, v8, s[36:39], 0 idxen
buffer_load_d16_b16 v153, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_hi_b16 v154, v7, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v155, v9, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v152, v8, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v153, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_b16 v158, v7, s[36:39], 0 idxen
buffer_load_d16_b16 v159, v9, s[36:39], 0 idxen
buffer_load_d16_b16 v156, v8, s[36:39], 0 idxen
buffer_load_d16_b16 v157, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_hi_b16 v158, v7, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v159, v9, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v156, v8, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v157, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
s_swappc_b64 s[64:65], s[64:65]
s_branch 4350
s_mov_b32 s56, 0xffffea64
s_bitcmp1_b32 s45, 4
s_cselect_b64 s[64:65], s[68:69], s[72:73]
s_bitcmp1_b32 s45, 2
s_cselect_b32 s56, s56, 0xffffeb90
s_setprio 2
_v_pk_mul_f16__vop3p 160, 410, 271, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 161, 414, 271, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 162, 408, 271, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 163, 412, 271, 0x0, 0x1, 0x0, 0x0
v_mov_b32 v154, v160 quad_perm:[1,0,3,2]
v_mov_b32 v158, v161 quad_perm:[1,0,3,2]
v_mov_b32 v152, v162 quad_perm:[1,0,3,2]
v_mov_b32 v156, v163 quad_perm:[1,0,3,2]
v_pk_fma_f16 v160, v154, v15, v160 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v161, v158, v15, v161 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v162, v152, v15, v162 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v163, v156, v15, v163 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_mov_b32 v154, v160 quad_perm:[2,3,0,1]
v_mov_b32 v158, v161 quad_perm:[2,3,0,1]
v_mov_b32 v152, v162 quad_perm:[2,3,0,1]
v_mov_b32 v156, v163 quad_perm:[2,3,0,1]
v_pk_fma_f16 v160, v154, v15, v160 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v161, v158, v15, v161 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v162, v152, v15, v162 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v163, v156, v15, v163 op_sel:[0,1,0] op_sel_hi:[1,1,1]
buffer_load_d16_b16 v154, v7, s[36:39], 0 idxen
buffer_load_d16_b16 v155, v9, s[36:39], 0 idxen
buffer_load_d16_b16 v152, v8, s[36:39], 0 idxen
buffer_load_d16_b16 v153, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_hi_b16 v154, v7, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v155, v9, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v152, v8, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v153, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_b16 v158, v7, s[36:39], 0 idxen
buffer_load_d16_b16 v159, v9, s[36:39], 0 idxen
buffer_load_d16_b16 v156, v8, s[36:39], 0 idxen
buffer_load_d16_b16 v157, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
buffer_load_d16_hi_b16 v158, v7, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v159, v9, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v156, v8, s[36:39], 0 idxen
buffer_load_d16_hi_b16 v157, v10, s[36:39], 0 idxen
s_sub_u32 s15, s15, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s59
s_addc_u32 s37, s37, 0
s_swappc_b64 s[64:65], s[64:65]
s_branch 4252
ds_store_b128 v1, v[18:21] offset:4672
ds_store_b128 v1, v[30:33] offset:16
s_setprio 1
s_ashr_i32 s57, s56, 31
s_add_u32 s64, s64, s56
s_addc_u32 s65, s65, s57
s_bitcmp1_b32 s45, 3
s_cselect_b64 vcc, -1, 0
s_bitcmp1_b32 s45, 2
s_cselect_b64 s[54:55], -1, 0
s_mov_b32 exec_hi, 0
s_waitcnt lgkmcnt(0)
s_barrier
v_readfirstlane_b32 s41, v4
v_mov_b32 v69, v36
v_mov_b32 v70, v37
v_mov_b32 v71, v38
v_mov_b32 v72, v39
v_mov_b32 v73, v40
v_mov_b32 v74, v41
v_mov_b32 v75, v42
v_mov_b32 v76, v43
_v_pk_add_f16__vop3p 88, 292, 317, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 89, 293, 318, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 90, 294, 319, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 91, 295, 320, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 92, 296, 321, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 93, 297, 322, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 94, 298, 323, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 95, 299, 324, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 88, 344, 240, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 89, 345, 240, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 90, 346, 240, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 91, 347, 240, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 92, 348, 240, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 93, 349, 240, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 94, 350, 240, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 95, 351, 240, 0x0, 0x1, 0x0, 0x0
v_pk_fma_f16 v88, v44, 0.5, v88 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v89, v45, 0.5, v89 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v90, v46, 0.5, v90 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v91, v47, 0.5, v91 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v92, v48, 0.5, v92 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v93, v49, 0.5, v93 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v94, v50, 0.5, v94 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v95, v51, 0.5, v95 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v105, v44, -1.0, v88 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v106, v45, -1.0, v89 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v107, v46, -1.0, v90 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v108, v47, -1.0, v91 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v109, v48, -1.0, v92 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v110, v49, -1.0, v93 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v111, v50, -1.0, v94 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v112, v51, -1.0, v95 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_mov_b32 v124, v61
v_mov_b32 v125, v62
v_mov_b32 v126, v63
v_mov_b32 v127, v64
v_mov_b32 v128, v65
v_mov_b32 v129, v66
v_mov_b32 v130, v67
v_mov_b32 v131, v68
s_mov_b32 exec_hi, -1
v_cndmask_b32 v11, v13, v1, vcc
v_cndmask_b32 v12, v14, v3, s[54:55]
s_bitcmp1_b32 s41, 1
s_addc_u32 s45, s45, s45
s_mov_b32 exec_hi, 0
ds_load_b128 v[36:39], v11 offset:27840
ds_load_b128 v[40:43], v11 offset:30144
ds_load_b128 v[44:47], v11 offset:32512
ds_load_b128 v[48:51], v11 offset:34816
s_mov_b32 exec_hi, -1
ds_store_b128 v12, v[160:163] offset:18560
s_mov_b32 exec_hi, 0
ds_load_b128 v[53:56], v11 offset:27856
ds_load_b128 v[57:60], v11 offset:30160
ds_load_b128 v[61:64], v11 offset:32528
ds_load_b128 v[65:68], v11 offset:34832
s_mov_b32 exec_hi, -1
ds_store_b128 v12, v[164:167] offset:19136
s_swappc_b64 s[64:65], s[64:65]
ds_store_b128 v2, v[18:21] offset:13952
ds_store_b128 v2, v[30:33] offset:9296
s_setprio 1
s_ashr_i32 s57, s56, 31
s_sub_u32 s23, s23, s34
s_cselect_b64 s[56:57], 0, s[56:57]
s_add_u32 s64, s64, s56
s_addc_u32 s65, s65, s57
s_bitcmp1_b32 s45, 3
s_cselect_b64 vcc, -1, 0
s_bitcmp1_b32 s45, 3
s_cselect_b64 s[54:55], -1, 0
s_mov_b32 exec_hi, 0
s_waitcnt lgkmcnt(0)
s_barrier
v_mov_b32 v77, v36
v_mov_b32 v78, v37
v_mov_b32 v79, v38
v_mov_b32 v80, v39
v_mov_b32 v81, v40
v_mov_b32 v82, v41
v_mov_b32 v83, v42
v_mov_b32 v84, v43
_v_pk_add_f16__vop3p 96, 292, 317, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 97, 293, 318, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 98, 294, 319, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 99, 295, 320, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 100, 296, 321, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 101, 297, 322, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 102, 298, 323, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 103, 299, 324, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 96, 352, 240, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 97, 353, 240, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 98, 354, 240, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 99, 355, 240, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 100, 356, 240, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 101, 357, 240, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 102, 358, 240, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 103, 359, 240, 0x0, 0x1, 0x0, 0x0
v_pk_fma_f16 v96, v44, 0.5, v96 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v97, v45, 0.5, v97 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v98, v46, 0.5, v98 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v99, v47, 0.5, v99 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v100, v48, 0.5, v100 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v101, v49, 0.5, v101 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v102, v50, 0.5, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v103, v51, 0.5, v103 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v113, v44, -1.0, v96 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v114, v45, -1.0, v97 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v115, v46, -1.0, v98 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v116, v47, -1.0, v99 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v117, v48, -1.0, v100 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v118, v49, -1.0, v101 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v119, v50, -1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v120, v51, -1.0, v103 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_mov_b32 v132, v61
v_mov_b32 v133, v62
v_mov_b32 v134, v63
v_mov_b32 v135, v64
v_mov_b32 v136, v65
v_mov_b32 v137, v66
v_mov_b32 v138, v67
v_mov_b32 v139, v68
s_mov_b32 exec_hi, -1
v_cndmask_b32 v11, v13, v2, vcc
v_cndmask_b32 v12, v14, v3, s[54:55]
s_bitcmp1_b32 s41, 0
s_cselect_b32 s35, 0, s35
s_cselect_b32 s34, 1, s34
s_lshr_b32 s39, s41, 16
ds_load_b128 v[7:10], v5 offset:37120
ds_load_b32 v4, v6 offset:39168
s_bitcmp1_b32 s41, 1
s_cselect_b32 s59, s49, s53
s_cselect_b64 s[36:37], s[16:17], s[18:19]
s_mul_i32 s56, s39, s59
s_mul_hi_u32 s57, s39, s59
s_add_u32 s15, s39, 1
s_sub_u32 s15, s22, s15
s_cselect_b32 s39, 0, s35
s_add_u32 s36, s36, s56
s_addc_u32 s37, s37, s57
s_mov_b32 exec_hi, 0
ds_load_b128 v[36:39], v11 offset:18560
ds_load_b128 v[40:43], v11 offset:20864
ds_load_b128 v[44:47], v11 offset:23232
ds_load_b128 v[48:51], v11 offset:25536
s_mov_b32 exec_hi, -1
ds_store_b128 v12, v[160:163] offset:27840
s_mov_b32 exec_hi, 0
ds_load_b128 v[53:56], v11 offset:18576
ds_load_b128 v[57:60], v11 offset:20880
ds_load_b128 v[61:64], v11 offset:23248
ds_load_b128 v[65:68], v11 offset:25552
s_mov_b32 exec_hi, -1
ds_store_b128 v12, v[164:167] offset:28416
s_waitcnt lgkmcnt(10)
s_swappc_b64 s[64:65], s[64:65]
ds_store_b128 v1, v[18:21] offset:4672
ds_store_b128 v1, v[30:33] offset:16
s_setprio 1
s_ashr_i32 s57, s56, 31
s_add_u32 s64, s64, s56
s_addc_u32 s65, s65, s57
s_bitcmp1_b32 s45, 3
s_cselect_b64 vcc, -1, 0
s_bitcmp1_b32 s45, 2
s_cselect_b64 s[54:55], -1, 0
s_mov_b32 exec_hi, 0
s_waitcnt lgkmcnt(0)
v_readfirstlane_b32 s41, v4
v_mov_b32 v69, v36
v_mov_b32 v70, v37
v_mov_b32 v71, v38
v_mov_b32 v72, v39
v_mov_b32 v73, v40
v_mov_b32 v74, v41
v_mov_b32 v75, v42
v_mov_b32 v76, v43
_v_pk_add_f16__vop3p 88, 292, 317, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 89, 293, 318, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 90, 294, 319, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 91, 295, 320, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 92, 296, 321, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 93, 297, 322, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 94, 298, 323, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 95, 299, 324, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 88, 344, 240, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 89, 345, 240, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 90, 346, 240, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 91, 347, 240, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 92, 348, 240, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 93, 349, 240, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 94, 350, 240, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 95, 351, 240, 0x0, 0x1, 0x0, 0x0
v_pk_fma_f16 v88, v44, 0.5, v88 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v89, v45, 0.5, v89 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v90, v46, 0.5, v90 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v91, v47, 0.5, v91 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v92, v48, 0.5, v92 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v93, v49, 0.5, v93 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v94, v50, 0.5, v94 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v95, v51, 0.5, v95 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v105, v44, -1.0, v88 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v106, v45, -1.0, v89 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v107, v46, -1.0, v90 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v108, v47, -1.0, v91 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v109, v48, -1.0, v92 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v110, v49, -1.0, v93 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v111, v50, -1.0, v94 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v112, v51, -1.0, v95 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_mov_b32 v124, v61
v_mov_b32 v125, v62
v_mov_b32 v126, v63
v_mov_b32 v127, v64
v_mov_b32 v128, v65
v_mov_b32 v129, v66
v_mov_b32 v130, v67
v_mov_b32 v131, v68
s_mov_b32 exec_hi, -1
v_cndmask_b32 v11, v13, v1, vcc
v_cndmask_b32 v12, v14, v3, s[54:55]
s_barrier
s_bitcmp1_b32 s41, 1
s_addc_u32 s45, s45, s45
s_mov_b32 exec_hi, 0
ds_load_b128 v[36:39], v11 offset:27840
ds_load_b128 v[40:43], v11 offset:30144
ds_load_b128 v[44:47], v11 offset:32512
ds_load_b128 v[48:51], v11 offset:34816
s_mov_b32 exec_hi, -1
ds_store_b128 v12, v[160:163] offset:18560
s_mov_b32 exec_hi, 0
ds_load_b128 v[53:56], v11 offset:27856
ds_load_b128 v[57:60], v11 offset:30160
ds_load_b128 v[61:64], v11 offset:32528
ds_load_b128 v[65:68], v11 offset:34832
s_mov_b32 exec_hi, -1
ds_store_b128 v12, v[164:167] offset:19136
s_swappc_b64 s[64:65], s[64:65]
ds_store_b128 v2, v[18:21] offset:13952
ds_store_b128 v2, v[30:33] offset:9296
s_setprio 1
s_ashr_i32 s57, s56, 31
s_sub_u32 s23, s23, s34
s_cselect_b64 s[56:57], 0, s[56:57]
s_add_u32 s64, s64, s56
s_addc_u32 s65, s65, s57
s_bitcmp1_b32 s45, 3
s_cselect_b64 vcc, -1, 0
s_bitcmp1_b32 s45, 3
s_cselect_b64 s[54:55], -1, 0
s_mov_b32 exec_hi, 0
s_waitcnt lgkmcnt(0)
v_mov_b32 v77, v36
v_mov_b32 v78, v37
v_mov_b32 v79, v38
v_mov_b32 v80, v39
v_mov_b32 v81, v40
v_mov_b32 v82, v41
v_mov_b32 v83, v42
v_mov_b32 v84, v43
_v_pk_add_f16__vop3p 96, 292, 317, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 97, 293, 318, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 98, 294, 319, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 99, 295, 320, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 100, 296, 321, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 101, 297, 322, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 102, 298, 323, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 103, 299, 324, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 96, 352, 240, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 97, 353, 240, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 98, 354, 240, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 99, 355, 240, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 100, 356, 240, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 101, 357, 240, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 102, 358, 240, 0x0, 0x1, 0x0, 0x0
_v_pk_mul_f16__vop3p 103, 359, 240, 0x0, 0x1, 0x0, 0x0
v_pk_fma_f16 v96, v44, 0.5, v96 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v97, v45, 0.5, v97 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v98, v46, 0.5, v98 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v99, v47, 0.5, v99 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v100, v48, 0.5, v100 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v101, v49, 0.5, v101 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v102, v50, 0.5, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v103, v51, 0.5, v103 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v113, v44, -1.0, v96 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v114, v45, -1.0, v97 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v115, v46, -1.0, v98 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v116, v47, -1.0, v99 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v117, v48, -1.0, v100 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v118, v49, -1.0, v101 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v119, v50, -1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v120, v51, -1.0, v103 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_mov_b32 v132, v61
v_mov_b32 v133, v62
v_mov_b32 v134, v63
v_mov_b32 v135, v64
v_mov_b32 v136, v65
v_mov_b32 v137, v66
v_mov_b32 v138, v67
v_mov_b32 v139, v68
s_mov_b32 exec_hi, -1
v_cndmask_b32 v11, v13, v2, vcc
v_cndmask_b32 v12, v14, v3, s[54:55]
s_barrier
s_bitcmp1_b32 s41, 0
s_cselect_b32 s35, 0, s35
s_cselect_b32 s34, 1, s34
s_lshr_b32 s39, s41, 16
ds_load_b128 v[7:10], v5 offset:37120
ds_load_b32 v4, v6 offset:39168
s_bitcmp1_b32 s41, 1
s_cselect_b32 s59, s49, s53
s_cselect_b64 s[36:37], s[16:17], s[18:19]
s_mul_i32 s56, s39, s59
s_mul_hi_u32 s57, s39, s59
s_add_u32 s15, s39, 1
s_sub_u32 s15, s22, s15
s_cselect_b32 s39, 0, s35
s_add_u32 s36, s36, s56
s_addc_u32 s37, s37, s57
s_mov_b32 exec_hi, 0
ds_load_b128 v[36:39], v11 offset:18560
ds_load_b128 v[40:43], v11 offset:20864
ds_load_b128 v[44:47], v11 offset:23232
ds_load_b128 v[48:51], v11 offset:25536
s_mov_b32 exec_hi, -1
ds_store_b128 v12, v[160:163] offset:27840
s_mov_b32 exec_hi, 0
ds_load_b128 v[53:56], v11 offset:18576
ds_load_b128 v[57:60], v11 offset:20880
ds_load_b128 v[61:64], v11 offset:23248
ds_load_b128 v[65:68], v11 offset:25552
s_mov_b32 exec_hi, -1
ds_store_b128 v12, v[164:167] offset:28416
s_waitcnt lgkmcnt(10)
s_swappc_b64 s[64:65], s[64:65]
ds_store_b128 v1, v[18:21] offset:4672
ds_store_b128 v1, v[30:33] offset:16
s_setprio 1
s_ashr_i32 s57, s56, 31
s_add_u32 s64, s64, s56
s_addc_u32 s65, s65, s57
s_bitcmp1_b32 s45, 3
s_cselect_b64 vcc, -1, 0
s_bitcmp1_b32 s45, 2
s_cselect_b64 s[54:55], -1, 0
s_mov_b32 exec_hi, 0
s_waitcnt lgkmcnt(0)
s_barrier
v_readfirstlane_b32 s41, v4
_v_pk_add_f16__vop3p 36, 292, 309, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 37, 293, 310, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 38, 294, 311, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 39, 295, 312, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 40, 296, 313, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 41, 297, 314, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 42, 298, 315, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 43, 299, 316, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 61, 317, 300, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 62, 318, 301, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 63, 319, 302, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 64, 320, 303, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 65, 321, 304, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 66, 322, 305, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 67, 323, 306, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 68, 324, 307, 0x0, 0x3, 0x2, 0x2
v_wmma_f16_16x16x16_f16 v[18:21], v[69:76], v[36:43], 0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_wmma_f16_16x16x16_f16 v[18:21], v[77:84], v[36:43], 0 op_sel:[0,0,1] op_sel_hi:[1,1,1]
v_wmma_f16_16x16x16_f16 v[30:33], v[124:131], v[61:68], 0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_wmma_f16_16x16x16_f16 v[30:33], v[132:139], v[61:68], 0 op_sel:[0,0,1] op_sel_hi:[1,1,1]
_v_pk_add_f16__vop3p 36, 300, 309, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 37, 301, 310, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 38, 302, 311, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 39, 303, 312, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 40, 304, 313, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 41, 305, 314, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 42, 306, 315, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 43, 307, 316, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 61, 309, 300, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 62, 310, 301, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 63, 311, 302, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 64, 312, 303, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 65, 313, 304, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 66, 314, 305, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 67, 315, 306, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 68, 316, 307, 0x0, 0x3, 0x2, 0x2
v_wmma_f16_16x16x16_f16 v[22:25], v[88:95], v[36:43], 0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
s_mov_b32 exec_hi, -1
v_wmma_f16_16x16x16_f16 v[22:25], v[96:103], v[36:43], 0 op_sel:[0,0,1] op_sel_hi:[1,1,1]
v_wmma_f16_16x16x16_f16 v[26:29], v[105:112], v[61:68], 0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_wmma_f16_16x16x16_f16 v[26:29], v[113:120], v[61:68], 0 op_sel:[0,0,1] op_sel_hi:[1,1,1]
_v_pk_add_f16__vop3p 18, 274, 278, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 19, 275, 279, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 20, 276, 280, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 21, 277, 281, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 30, 278, 286, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 31, 279, 287, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 32, 280, 288, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 33, 281, 289, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 18, 274, 282, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 19, 275, 283, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 20, 276, 284, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 21, 277, 285, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 30, 286, 282, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 31, 287, 283, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 32, 288, 284, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 33, 289, 285, 0x0, 0x3, 0x2, 0x2
v_cndmask_b32 v11, v13, v1, vcc
v_cndmask_b32 v12, v14, v3, s[54:55]
s_bitcmp1_b32 s41, 1
s_addc_u32 s45, s45, s45
s_mov_b32 exec_hi, 0
ds_load_b128 v[36:39], v11 offset:27840
ds_load_b128 v[40:43], v11 offset:30144
ds_load_b128 v[44:47], v11 offset:32512
ds_load_b128 v[48:51], v11 offset:34816
s_mov_b32 exec_hi, -1
ds_store_b128 v12, v[160:163] offset:18560
s_mov_b32 exec_hi, 0
ds_load_b128 v[53:56], v11 offset:27856
ds_load_b128 v[57:60], v11 offset:30160
ds_load_b128 v[61:64], v11 offset:32528
ds_load_b128 v[65:68], v11 offset:34832
s_mov_b32 exec_hi, -1
ds_store_b128 v12, v[164:167] offset:19136
s_swappc_b64 s[64:65], s[64:65]
ds_store_b128 v2, v[18:21] offset:13952
ds_store_b128 v2, v[30:33] offset:9296
s_setprio 1
s_ashr_i32 s57, s56, 31
s_sub_u32 s23, s23, s34
s_cselect_b64 s[56:57], 0, s[56:57]
s_add_u32 s64, s64, s56
s_addc_u32 s65, s65, s57
s_bitcmp1_b32 s45, 3
s_cselect_b64 vcc, -1, 0
s_bitcmp1_b32 s45, 3
s_cselect_b64 s[54:55], -1, 0
s_mov_b32 exec_hi, 0
s_waitcnt lgkmcnt(0)
s_barrier
_v_pk_add_f16__vop3p 36, 292, 309, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 37, 293, 310, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 38, 294, 311, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 39, 295, 312, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 40, 296, 313, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 41, 297, 314, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 42, 298, 315, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 43, 299, 316, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 61, 317, 300, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 62, 318, 301, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 63, 319, 302, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 64, 320, 303, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 65, 321, 304, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 66, 322, 305, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 67, 323, 306, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 68, 324, 307, 0x0, 0x3, 0x2, 0x2
v_wmma_f16_16x16x16_f16 v[18:21], v[69:76], v[36:43], 0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_wmma_f16_16x16x16_f16 v[18:21], v[77:84], v[36:43], 0 op_sel:[0,0,1] op_sel_hi:[1,1,1]
v_wmma_f16_16x16x16_f16 v[30:33], v[124:131], v[61:68], 0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_wmma_f16_16x16x16_f16 v[30:33], v[132:139], v[61:68], 0 op_sel:[0,0,1] op_sel_hi:[1,1,1]
_v_pk_add_f16__vop3p 36, 300, 309, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 37, 301, 310, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 38, 302, 311, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 39, 303, 312, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 40, 304, 313, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 41, 305, 314, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 42, 306, 315, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 43, 307, 316, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 61, 309, 300, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 62, 310, 301, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 63, 311, 302, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 64, 312, 303, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 65, 313, 304, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 66, 314, 305, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 67, 315, 306, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 68, 316, 307, 0x0, 0x3, 0x2, 0x2
v_wmma_f16_16x16x16_f16 v[22:25], v[88:95], v[36:43], 0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
s_mov_b32 exec_hi, -1
v_wmma_f16_16x16x16_f16 v[22:25], v[96:103], v[36:43], 0 op_sel:[0,0,1] op_sel_hi:[1,1,1]
v_wmma_f16_16x16x16_f16 v[26:29], v[105:112], v[61:68], 0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_wmma_f16_16x16x16_f16 v[26:29], v[113:120], v[61:68], 0 op_sel:[0,0,1] op_sel_hi:[1,1,1]
_v_pk_add_f16__vop3p 18, 274, 278, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 19, 275, 279, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 20, 276, 280, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 21, 277, 281, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 30, 278, 286, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 31, 279, 287, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 32, 280, 288, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 33, 281, 289, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 18, 274, 282, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 19, 275, 283, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 20, 276, 284, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 21, 277, 285, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 30, 286, 282, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 31, 287, 283, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 32, 288, 284, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 33, 289, 285, 0x0, 0x3, 0x2, 0x2
v_cndmask_b32 v11, v13, v2, vcc
v_cndmask_b32 v12, v14, v3, s[54:55]
s_bitcmp1_b32 s41, 0
s_cselect_b32 s35, 0, s35
s_cselect_b32 s34, 1, s34
s_lshr_b32 s39, s41, 16
ds_load_b128 v[7:10], v5 offset:37120
ds_load_b32 v4, v6 offset:39168
s_bitcmp1_b32 s41, 1
s_cselect_b32 s59, s49, s53
s_cselect_b64 s[36:37], s[16:17], s[18:19]
s_mul_i32 s56, s39, s59
s_mul_hi_u32 s57, s39, s59
s_add_u32 s15, s39, 1
s_sub_u32 s15, s22, s15
s_cselect_b32 s39, 0, s35
s_add_u32 s36, s36, s56
s_addc_u32 s37, s37, s57
s_mov_b32 exec_hi, 0
ds_load_b128 v[36:39], v11 offset:18560
ds_load_b128 v[40:43], v11 offset:20864
ds_load_b128 v[44:47], v11 offset:23232
ds_load_b128 v[48:51], v11 offset:25536
s_mov_b32 exec_hi, -1
ds_store_b128 v12, v[160:163] offset:27840
s_mov_b32 exec_hi, 0
ds_load_b128 v[53:56], v11 offset:18576
ds_load_b128 v[57:60], v11 offset:20880
ds_load_b128 v[61:64], v11 offset:23248
ds_load_b128 v[65:68], v11 offset:25552
s_mov_b32 exec_hi, -1
ds_store_b128 v12, v[164:167] offset:28416
s_waitcnt lgkmcnt(10)
s_swappc_b64 s[64:65], s[64:65]
ds_store_b128 v1, v[18:21] offset:4672
ds_store_b128 v1, v[30:33] offset:16
s_setprio 1
s_ashr_i32 s57, s56, 31
s_add_u32 s64, s64, s56
s_addc_u32 s65, s65, s57
s_bitcmp1_b32 s45, 3
s_cselect_b64 vcc, -1, 0
s_bitcmp1_b32 s45, 2
s_cselect_b64 s[54:55], -1, 0
s_mov_b32 exec_hi, 0
s_waitcnt lgkmcnt(0)
v_readfirstlane_b32 s41, v4
_v_pk_add_f16__vop3p 36, 292, 309, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 37, 293, 310, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 38, 294, 311, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 39, 295, 312, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 40, 296, 313, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 41, 297, 314, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 42, 298, 315, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 43, 299, 316, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 61, 317, 300, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 62, 318, 301, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 63, 319, 302, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 64, 320, 303, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 65, 321, 304, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 66, 322, 305, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 67, 323, 306, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 68, 324, 307, 0x0, 0x3, 0x2, 0x2
v_wmma_f16_16x16x16_f16 v[18:21], v[69:76], v[36:43], 0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_wmma_f16_16x16x16_f16 v[18:21], v[77:84], v[36:43], 0 op_sel:[0,0,1] op_sel_hi:[1,1,1]
v_wmma_f16_16x16x16_f16 v[30:33], v[124:131], v[61:68], 0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_wmma_f16_16x16x16_f16 v[30:33], v[132:139], v[61:68], 0 op_sel:[0,0,1] op_sel_hi:[1,1,1]
_v_pk_add_f16__vop3p 36, 300, 309, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 37, 301, 310, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 38, 302, 311, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 39, 303, 312, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 40, 304, 313, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 41, 305, 314, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 42, 306, 315, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 43, 307, 316, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 61, 309, 300, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 62, 310, 301, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 63, 311, 302, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 64, 312, 303, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 65, 313, 304, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 66, 314, 305, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 67, 315, 306, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 68, 316, 307, 0x0, 0x3, 0x2, 0x2
v_wmma_f16_16x16x16_f16 v[22:25], v[88:95], v[36:43], 0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
s_mov_b32 exec_hi, -1
v_wmma_f16_16x16x16_f16 v[22:25], v[96:103], v[36:43], 0 op_sel:[0,0,1] op_sel_hi:[1,1,1]
v_wmma_f16_16x16x16_f16 v[26:29], v[105:112], v[61:68], 0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_wmma_f16_16x16x16_f16 v[26:29], v[113:120], v[61:68], 0 op_sel:[0,0,1] op_sel_hi:[1,1,1]
_v_pk_add_f16__vop3p 18, 274, 278, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 19, 275, 279, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 20, 276, 280, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 21, 277, 281, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 30, 278, 286, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 31, 279, 287, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 32, 280, 288, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 33, 281, 289, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 18, 274, 282, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 19, 275, 283, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 20, 276, 284, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 21, 277, 285, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 30, 286, 282, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 31, 287, 283, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 32, 288, 284, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 33, 289, 285, 0x0, 0x3, 0x2, 0x2
v_cndmask_b32 v11, v13, v1, vcc
v_cndmask_b32 v12, v14, v3, s[54:55]
s_barrier
s_bitcmp1_b32 s41, 1
s_addc_u32 s45, s45, s45
s_mov_b32 exec_hi, 0
ds_load_b128 v[36:39], v11 offset:27840
ds_load_b128 v[40:43], v11 offset:30144
ds_load_b128 v[44:47], v11 offset:32512
ds_load_b128 v[48:51], v11 offset:34816
s_mov_b32 exec_hi, -1
ds_store_b128 v12, v[160:163] offset:18560
s_mov_b32 exec_hi, 0
ds_load_b128 v[53:56], v11 offset:27856
ds_load_b128 v[57:60], v11 offset:30160
ds_load_b128 v[61:64], v11 offset:32528
ds_load_b128 v[65:68], v11 offset:34832
s_mov_b32 exec_hi, -1
ds_store_b128 v12, v[164:167] offset:19136
s_swappc_b64 s[64:65], s[64:65]
ds_store_b128 v2, v[18:21] offset:13952
ds_store_b128 v2, v[30:33] offset:9296
s_setprio 1
s_ashr_i32 s57, s56, 31
s_sub_u32 s23, s23, s34
s_cselect_b64 s[56:57], 0, s[56:57]
s_add_u32 s64, s64, s56
s_addc_u32 s65, s65, s57
s_bitcmp1_b32 s45, 3
s_cselect_b64 vcc, -1, 0
s_bitcmp1_b32 s45, 3
s_cselect_b64 s[54:55], -1, 0
s_mov_b32 exec_hi, 0
s_waitcnt lgkmcnt(0)
_v_pk_add_f16__vop3p 36, 292, 309, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 37, 293, 310, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 38, 294, 311, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 39, 295, 312, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 40, 296, 313, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 41, 297, 314, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 42, 298, 315, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 43, 299, 316, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 61, 317, 300, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 62, 318, 301, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 63, 319, 302, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 64, 320, 303, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 65, 321, 304, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 66, 322, 305, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 67, 323, 306, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 68, 324, 307, 0x0, 0x3, 0x2, 0x2
v_wmma_f16_16x16x16_f16 v[18:21], v[69:76], v[36:43], 0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_wmma_f16_16x16x16_f16 v[18:21], v[77:84], v[36:43], 0 op_sel:[0,0,1] op_sel_hi:[1,1,1]
v_wmma_f16_16x16x16_f16 v[30:33], v[124:131], v[61:68], 0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_wmma_f16_16x16x16_f16 v[30:33], v[132:139], v[61:68], 0 op_sel:[0,0,1] op_sel_hi:[1,1,1]
_v_pk_add_f16__vop3p 36, 300, 309, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 37, 301, 310, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 38, 302, 311, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 39, 303, 312, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 40, 304, 313, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 41, 305, 314, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 42, 306, 315, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 43, 307, 316, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 61, 309, 300, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 62, 310, 301, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 63, 311, 302, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 64, 312, 303, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 65, 313, 304, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 66, 314, 305, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 67, 315, 306, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 68, 316, 307, 0x0, 0x3, 0x2, 0x2
v_wmma_f16_16x16x16_f16 v[22:25], v[88:95], v[36:43], 0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
s_mov_b32 exec_hi, -1
v_wmma_f16_16x16x16_f16 v[22:25], v[96:103], v[36:43], 0 op_sel:[0,0,1] op_sel_hi:[1,1,1]
v_wmma_f16_16x16x16_f16 v[26:29], v[105:112], v[61:68], 0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_wmma_f16_16x16x16_f16 v[26:29], v[113:120], v[61:68], 0 op_sel:[0,0,1] op_sel_hi:[1,1,1]
_v_pk_add_f16__vop3p 18, 274, 278, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 19, 275, 279, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 20, 276, 280, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 21, 277, 281, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 30, 278, 286, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 31, 279, 287, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 32, 280, 288, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 33, 281, 289, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 18, 274, 282, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 19, 275, 283, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 20, 276, 284, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 21, 277, 285, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 30, 286, 282, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 31, 287, 283, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 32, 288, 284, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 33, 289, 285, 0x0, 0x3, 0x2, 0x2
v_cndmask_b32 v11, v13, v2, vcc
v_cndmask_b32 v12, v14, v3, s[54:55]
s_barrier
s_bitcmp1_b32 s41, 0
s_cselect_b32 s35, 0, s35
s_cselect_b32 s34, 1, s34
s_lshr_b32 s39, s41, 16
ds_load_b128 v[7:10], v5 offset:37120
ds_load_b32 v4, v6 offset:39168
s_bitcmp1_b32 s41, 1
s_cselect_b32 s59, s49, s53
s_cselect_b64 s[36:37], s[16:17], s[18:19]
s_mul_i32 s56, s39, s59
s_mul_hi_u32 s57, s39, s59
s_add_u32 s15, s39, 1
s_sub_u32 s15, s22, s15
s_cselect_b32 s39, 0, s35
s_add_u32 s36, s36, s56
s_addc_u32 s37, s37, s57
s_mov_b32 exec_hi, 0
ds_load_b128 v[36:39], v11 offset:18560
ds_load_b128 v[40:43], v11 offset:20864
ds_load_b128 v[44:47], v11 offset:23232
ds_load_b128 v[48:51], v11 offset:25536
s_mov_b32 exec_hi, -1
ds_store_b128 v12, v[160:163] offset:27840
s_mov_b32 exec_hi, 0
ds_load_b128 v[53:56], v11 offset:18576
ds_load_b128 v[57:60], v11 offset:20880
ds_load_b128 v[61:64], v11 offset:23248
ds_load_b128 v[65:68], v11 offset:25552
s_mov_b32 exec_hi, -1
ds_store_b128 v12, v[164:167] offset:28416
s_waitcnt lgkmcnt(10)
s_swappc_b64 s[64:65], s[64:65]
v_bfe_u32 v21, v0, 6, 1
v_and_b32 v16, 63, v0
v_cmp_eq_u32 vcc, v21, 1
v_cndmask_b32 v23, 0, 0x800, vcc
v_cndmask_b32 v21, 0, 0x400, vcc
v_cndmask_b32 v22, 0, 0x100, vcc
v_lshl_add_u32 v14, v16, 3, v23
v_lshl_add_u32 v17, v16, 2, v22
v_lshl_add_u32 v18, v16, 2, 0
v_lshl_add_u32 v16, v16, 4, v21
s_cmp_eq_u64 s[30:31], 0
s_cselect_b32 s91, 0, 0x11014000
s_and_b32 s31, s31, 0xffff
s_add_u32 s31, s31, 0x20000
s_mov_b64 s[88:89], s[30:31]
s_mov_b32 s90, 0x80000000
v_and_b32 v21, v0, 63
v_lshlrev_b32 v21, 1, v21
v_cmp_lt_u32 vcc, v21, s12
v_add_nc_u32 v22, v21, 1
v_cndmask_b32 v21, 0x80000000, v21, vcc
v_cmp_lt_u32 vcc, v22, s12
v_cndmask_b32 v22, 0x80000000, v22, vcc
buffer_load_d16_b16 v23, v21, s[88:91], 0 idxen
buffer_load_d16_hi_b16 v23, v22, s[88:91], 0 idxen
s_waitcnt vmcnt(0)
v_readlane_b32 s56, v23, 0
v_readlane_b32 s57, v23, 1
v_readlane_b32 s59, v23, 2
v_readlane_b32 s64, v23, 3
v_readlane_b32 s65, v23, 4
v_readlane_b32 s66, v23, 5
v_readlane_b32 s67, v23, 6
v_readlane_b32 s68, v23, 7
s_bfe_u32 s88, s58, 0x80000
s_cmp_eq_u32 s88, 2
s_cbranch_scc1 20
s_cmp_eq_u32 s88, 0
s_cselect_b32 s32, 1.0, s32
v_cvt_f16_f32 v21, s32
v_readfirstlane_b32 s32, v21
v_cvt_f16_f32 v21, s33
v_readfirstlane_b32 s33, v21
_v_cmp_gt_f16__vop3_s_lit 106, 32, 0x3c00, 0x0, 0x0
s_pack_ll_b32_b16 s32, s32, s32
s_pack_ll_b32_b16 s33, s33, s33
s_cmp_eq_u32 s88, 3
s_cbranch_scc1 10
s_cbranch_vccnz 3
s_mov_b32 s84, 0x424c
s_branch 8
s_mov_b32 s84, 0x45e4
s_branch 5
s_mov_b32 s84, 0x497c
s_branch 2
s_mov_b32 s84, 0x4f94
s_add_u32 s86, s6, 0x3c90
s_addc_u32 s87, s7, 0
s_mov_b32 s82, 0xbc00c000
s_mov_b32 s40, 0x10000
s_mov_b32 s41, 0x30002
s_mov_b32 s45, 0x10000
v_readfirstlane_b32 s88, v0
s_and_b32 null, 64, s88
s_cmov_b32 s82, 0x3c00c000
s_cmov_b32 s40, 0x20003
s_cmov_b32 s41, 1
s_cmov_b32 s45, 1
s_and_b32 s21, s21, 0xffff
s_add_u32 s21, s21, 0x20000
s_lshl_b32 s80, s51, 1
s_lshl_b32 s81, s52, 1
s_mov_b64 s[72:73], s[20:21]
s_mov_b32 s74, 0x80000000
s_mov_b32 s75, 0
s_sub_u32 s89, s25, 1
s_bitcmp1_b32 s14, 1
s_cselect_b32 s89, s89, 0
s_cselect_b32 s88, -1, 1
s_sub_u32 s91, s24, 1
s_bitcmp1_b32 s14, 0
s_cselect_b32 s91, s91, 0
s_cselect_b32 s90, -1, 1
v_bfe_u32 v24, v0, 6, 1
v_bfe_u32 v25, v0, 4, 1
v_bfe_u32 v21, v0, 5, 1
v_lshl_add_u32 v24, v24, 2, 0
v_lshl_add_u32 v25, v25, 3, v24
v_bfe_u32 v23, v0, 2, 2
v_bfe_u32 v24, v0, 3, 1
v_xor_b32 v22, v0, v0 quad_perm:[0,0,3,1]
v_lshl_add_u32 v21, v21, 1, v25
v_xor_b32 v23, v23, v24
v_add_nc_u32 v24, v21, 1
v_mad_i32_i16 v19, v23, s88, s89 op_sel:[0,0,0,0]
v_mad_i32_i16 v25, v22, s90, s91 op_sel:[0,0,0,0]
v_mad_u32_u16 v19, v25, s48, v19 op_sel:[0,0,0,0]
v_cmp_lt_u32 vcc, v23, s25
v_cndmask_b32 v19, 0x80000000, v19, vcc
v_cmp_lt_u32 vcc, v22, s24
v_cndmask_b32 v19, 0x80000000, v19, vcc
v_mad_u32_u24 v20, v24, s46, v19
v_mad_u32_u24 v19, v21, s46, v19
v_cmp_lt_u32 vcc, v24, s12
v_cndmask_b32 v20, 0x80000000, v20, vcc
v_cmp_lt_u32 vcc, v21, s12
v_cndmask_b32 v19, 0x80000000, v19, vcc
s_add_u32 s89, s28, 1
s_lshr_b32 s89, s89, 1
s_lshl_b32 s90, s89, 1
s_add_u32 s91, s29, 1
s_lshr_b32 s91, s91, 1
s_lshl1_add_u32 s91, s91, 2
s_pack_ll_b32_b16 s22, s91, s89
s_pack_ll_b32_b16 s34, s11, s10
s_sub_u32 s35, s90, s26
s_sub_u32 s88, s91, s27
s_pack_ll_b32_b16 s35, s88, s35
s_pack_ll_b32_b16 s37, s29, s28
s_sub_u32 s88, s91, 1
s_pack_ll_b32_b16 s38, s88, s90
v_lshrrev_b32 v24, 16, s22
v_bfi_b32 v25, 0xffff, s22, 0
v_and_b32 v27, 1, v0
v_bfe_u32 v33, v0, 6, 1
v_and_b32 v22, 63, v0
v_mad_u32_u16 v28, 0x7c, s1, 0 op_sel:[0,0,0,0]
v_mad_u32_u16 v33, 2, s5, v33 op_sel:[0,0,0,0]
v_mad_u32_u16 v26, v24, v25, 0 op_sel:[0,0,0,0]
v_cmp_eq_u32 vcc, 0, v27
v_cndmask_b32 v34, v26, v25, vcc
v_mad_u32_u16 v23, 62, v33, v22 op_sel:[0,0,0,0]
v_cndmask_b32 v23, v28, v23, vcc
v_clz_i32_u32 v40, v34
v_lshlrev_b32 v41, v40, v34
v_and_b32 v39, 0xffffff00, v41
v_cmp_eq_u32 vcc, 0x80000000, v41
v_cvt_f32_u32 v39, v39
v_rcp_f32 v35, v39
v_sub_co_ci_u32 v36, vcc, 32, v40, vcc
v_cvt_f32_ubyte0 v40, v41
v_fma_f32 v39, v39, v35, -1.0
v_fma_f32 v39, v40, v35, v39
v_fmaak_f32 v39, v39, v35, 0x9f000000
v_mul_f32 v39, 0x5f800000, v39
v_mov_b32 v40, 0
v_cvt_floor_i32_f32 v39, -v39
v_lshl_add_u32 v35, v35, 9, v39
v_mad_u64_u32 v[40:41], vcc, v41, v35, v[40:41]
v_sub_co_ci_u32 v35, vcc, v35, -1, vcc
v_mov_b32 v38, v36 quad_perm:[1,1,1,1]
v_mov_b32 v36, v36 quad_perm:[0,0,0,0]
v_mov_b32 v37, v35 quad_perm:[1,1,1,1]
v_mov_b32 v35, v35 quad_perm:[0,0,0,0]
v_mul_hi_u32 v39, v23, v35
v_add_co_u32 v21, vcc, v39, v23
v_add_co_ci_u32 v39, vcc, 0, 0, vcc
v_cmp_eq_u32 vcc, 32, v36
v_cndmask_b32 v21, v21, v39, vcc
v_alignbit_b32 v21, v39, v21, v36
v_mul_hi_u32 v39, v23, v37
v_add_co_u32 v4, vcc, v39, v23
v_add_co_ci_u32 v39, vcc, 0, 0, vcc
v_cmp_eq_u32 vcc, 32, v38
v_cndmask_b32 v4, v4, v39, vcc
v_alignbit_b32 v4, v39, v4, v38
v_mad_u32_u16 v32, v21, v25, 0 op_sel:[0,0,0,0]
v_mad_u32_u16 v31, v4, v24, 0 op_sel:[0,0,0,0]
v_sub_nc_u32 v32, v23, v32
v_sub_nc_u32 v31, v21, v31
v_readlane_b32 s92, v32, 1
v_sub_nc_u32 v32, v32, v25
v_readlane_b32 s23, v31, 1
v_sub_nc_u32 v31, v31, v24
v_readlane_b32 s15, v4, 1
v_sub_nc_u32 v4, v4, s8
s_lshl_b32 s23, s23, 16
s_and_b32 s92, s92, 0xffff
s_add_u32 s23, s23, s92
v_mov_b32 v32, v32 quad_perm:[0,0,2,2]
v_mov_b32 v31, v31 quad_perm:[0,0,2,2]
v_mov_b32 v4, v4 quad_perm:[0,0,2,2]
v_add_co_u32 v32, vcc, v32, v27
v_cndmask_b32 v30, 0, v25, vcc
v_add_co_ci_u32 v31, vcc, v31, 0, vcc
v_cndmask_b32 v29, 0, v24, vcc
v_add_co_ci_u32 v4, vcc, v4, 0, vcc
v_min_u32 v27, v22, 63
v_sub_nc_u32 v32, v32, v30
v_sub_nc_u32 v31, v31, v29
v_cmp_eq_u32 vcc, v22, v27
v_lshlrev_b32 v5, 16, v31
v_bfi_b32 v5, 0xffff, v32, v5
v_add_nc_u32 v42, v4, s8
v_med3_u32 v27, v22, 1, 62
v_mul_lo_u32 v6, v42, s42
v_mul_lo_u32 v11, v42, s50
s_mul_i32 s36, s15, s42
s_mul_i32 s39, s15, s50
v_cndmask_b32 v6, 0x80000000, v6, vcc
v_cmp_eq_u32 vcc, v22, v27
v_cndmask_b32 v11, 0x80000000, v11, vcc
v_cmp_ge_u32 s[54:55], v42, s8
v_cndmask_b32 v6, v6, 0x80000000, s[54:55]
v_cndmask_b32 v11, v11, 0x80000000, s[54:55]
s_mov_b32 s49, 3
s_lshl_b32 s53, s49, 9
v_add_nc_u32 v15, s53, v14
s_bfe_u32 s10, s58, 0x80008
s_bfe_u32 s11, s58, 0x80010
s_cmp_eq_u32 s11, 0
s_cmov_b32 s26, 0
s_cbranch_scc1 108
s_add_u32 s11, s11, 0xffffff00
s_add_u32 s60, s60, 0
s_addc_u32 s61, s61, 0
s_lshr_b32 s91, s13, 2
s_or_b32 s91, s91, 0x21010000
v_cmp_eq_u32 vcc, v0, 0x100
s_cmp_eq_u64 vcc, 0
s_cselect_b32 s91, 0, s91
s_cselect_b32 s90, 0, 0x1010101
s_sub_u32 s10, 0, s10
s_mov_b64 s[88:89], s[60:61]
s_and_b32 s89, s89, 0xffff
s_or_b32 s89, s89, 0x40000
s_and_b32 s29, s22, 0xffff
s_lshr_b32 s28, s22, 16
s_lshr_b32 s29, s29, 1
s_mul_i32 s27, s29, s28
s_mul_i32 s27, s27, s8
s_add_u32 s27, s27, 61
v_writelane_b32 v22, 62, 0
v_writelane_b32 v22, s1, 1
v_writelane_b32 v22, 5, 2
v_clz_i32_u32 v26, v22
v_lshlrev_b32 v27, v26, v22
v_and_b32 v28, 0xffffff00, v27
v_cmp_eq_u32 vcc, 0x80000000, v27
v_cvt_f32_u32 v28, v28
v_rcp_f32 v24, v28
v_sub_co_ci_u32 v25, vcc, 32, v26, vcc
v_cvt_f32_ubyte0 v26, v27
v_fma_f32 v28, v28, v24, -1.0
v_fma_f32 v28, v26, v24, v28
v_fmaak_f32 v28, v28, v24, 0x9f000000
v_mul_f32 v28, 0x5f800000, v28
v_mov_b32 v26, 0
v_cvt_floor_i32_f32 v28, -v28
v_lshl_add_u32 v24, v24, 9, v28
v_mad_u64_u32 v[26:27], vcc, v27, v24, v[26:27]
v_sub_co_ci_u32 v24, vcc, v24, -1, vcc
v_mul_hi_u32 v26, s27, v24
v_add_co_u32 v23, vcc, v26, s27
v_add_co_ci_u32 v26, vcc, 0, 0, vcc
v_cmp_eq_u32 vcc, 32, v25
v_cndmask_b32 v23, v23, v26, vcc
v_alignbit_b32 v23, v26, v23, v25
v_mov_b32 v23, v23 quad_perm:[0,0,0,0]
v_mul_hi_u32 v26, v23, v24
v_add_co_u32 v22, vcc, v26, v23
v_add_co_ci_u32 v26, vcc, 0, 0, vcc
v_cmp_eq_u32 vcc, 32, v25
v_cndmask_b32 v22, v22, v26, vcc
v_alignbit_b32 v22, v26, v22, v25
v_mov_b32 v22, v22 quad_perm:[1,1,1,1]
v_add_nc_u32 v23, v22, 4
v_mul_hi_u32 v26, v23, v24
v_add_co_u32 v23, vcc, v26, v23
v_add_co_ci_u32 v26, vcc, 0, 0, vcc
v_cmp_eq_u32 vcc, 32, v25
v_cndmask_b32 v23, v23, v26, vcc
v_alignbit_b32 v23, v26, v23, v25
v_readlane_b32 s28, v22, 1
v_readlane_b32 s29, v23, 2
s_add_u32 s27, s9, 15
s_lshr_b32 s27, s27, 4
s_cmp_eq_u32 s27, 1
s_cmov_b32 s29, 1
s_add_u32 s26, s28, s29
s_mul_i32 s26, s27, s26
s_add_u32 s26, 6, s26
s_sub_u32 s26, s26, 1
s_mov_b32 s92, 0
s_mov_b32 s93, 0
s_mov_b32 s94, 0
s_mov_b32 s95, 0
s_mov_b32 s96, 0
s_mov_b32 s97, 0
s_mov_b32 s28, 0
s_mov_b32 s27, 8
s_cmp_gt_u32 s28, 0
s_cbranch_scc1 4
v_mov_b32 v58, v4
v_mov_b32 v63, v5
v_mov_b32 v145, v6
v_mov_b32 v146, v11
v_mov_b32 v4, v58
v_mov_b32 v5, v63
v_mov_b32 v6, v145
v_mov_b32 v11, v146
s_add_u32 s28, s28, 16
s_cmp_ge_u32 s28, s9
s_cmov_b32 s28, 0
s_cselect_b32 s29, 6, 2
s_cselect_b32 s98, 9, 0
s_pack_lh_b32_b16 s29, s29, s27
s_pack_ll_b32_b16 s98, s98, s28
v_mov_b32 v144, s29
s_swappc_b64 s[86:87], s[86:87]
s_waitcnt lgkmcnt(0)
s_barrier
v_pk_fma_f16 v44, v49, s82, v44 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v45, v50, s82, v45 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v46, v51, s82, v46 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v47, v52, s82, v47 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_mov_b32 v7, v19
v_mov_b32 v8, v20
v_mov_b32 v9, 0x80000000
v_mov_b32 v10, 0x80000000
v_mov_b32 v12, 0x80000000
v_mov_b32 v13, 0x80000000
s_setprio 0
ds_load_b128 v[34:37], v3
ds_store_b128 v16, v[7:10] offset:37120
ds_load_b128 v[39:42], v3 offset:576
ds_store_b32 v17, v144 offset:39168
s_setprio 2
s_sub_u32 s26, s26, 1
s_cselect_b32 s91, 0x21010000, s91
s_bitcmp1_b32 s92, 2
s_cselect_b32 s86, s84, 0x3c90
s_add_u32 s86, s6, s86
s_addc_u32 s87, s7, 0
s_swappc_b64 s[86:87], s[86:87]
s_waitcnt lgkmcnt(0)
v_add_nc_u32 v15, s53, v14
v_mov_b32 v165, v163
v_mov_b32 v166, v164
v_pk_fma_f16 v147, v34, s82, v24 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v148, v35, s82, v25 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v149, v36, s82, v26 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v150, v37, s82, v27 op_sel:[0,1,0] op_sel_hi:[1,1,1]
_v_pk_add_f16__vop3p 34, 285, 290, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 35, 286, 291, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 36, 287, 292, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 37, 288, 293, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 151, 290, 295, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 152, 291, 296, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 153, 292, 297, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 154, 293, 298, 0x0, 0x3, 0x0, 0x0
s_setprio 0
ds_load_b64 v[163:164], v15 offset:39680
ds_load_b128 v[54:57], v3 offset:2304
ds_load_b128 v[59:62], v3 offset:2880
s_setprio 2
s_mov_b32 s92, s93
s_mov_b32 s93, s94
s_mov_b32 s94, s95
s_mov_b32 s95, s96
s_mov_b32 s96, s97
s_mov_b32 s97, s27
s_bitcmp1_b32 s92, 0
s_cbranch_scc1 2473
s_sub_u32 s49, s49, 1
s_cselect_b32 s49, 3, s49
s_lshl_b32 s53, s49, 9
s_bitcmp1_b32 s92, 1
s_cselect_b32 s86, s85, 0x3c94
s_add_u32 s86, s6, s86
s_addc_u32 s87, s7, 0
s_bitcmp1_b32 s92, 2
s_cselect_b32 s75, 0x11014000, 0
s_sub_u32 s69, s12, 1
s_cselect_b32 s75, 0, s75
s_mov_b64 s[72:73], s[20:21]
s_swappc_b64 s[86:87], s[86:87]
s_waitcnt lgkmcnt(0)
s_barrier
v_pk_fma_f16 v155, v54, s82, v44 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v156, v55, s82, v45 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v157, v56, s82, v46 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v158, v57, s82, v47 op_sel:[0,1,0] op_sel_hi:[1,1,1]
_v_pk_add_f16__vop3p 54, 305, 310, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 55, 306, 311, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 56, 307, 312, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 57, 308, 313, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 159, 310, 315, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 160, 311, 316, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 161, 312, 317, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 162, 313, 318, 0x0, 0x3, 0x0, 0x0
s_add_u32 s11, s11, 0x100
s_cbranch_scc0 7
s_bitset0_b32 s91, 23
s_lshl_b64 exec, 1, s90
buffer_store_b8 v0, off, s[88:91], s4
s_mov_b64 exec, -1
s_mul_i32 s11, s11, 0xffffff01
s_and_not1_b32 null, 0xffffff00, s11
s_cbranch_scc1 3
s_bitset1_b32 s91, 23
buffer_load_b32 v21, off, s[88:91], null glc
s_setprio 0
s_nop 1
ds_load_b128 v[24:27], v3 offset:9280
ds_store_b64 v15, v[12:13] offset:39680
ds_load_b128 v[29:32], v3 offset:9856
ds_load_b32 v144, v18 offset:39168
s_setprio 2
s_bitcmp1_b32 s92, 2
s_cselect_b32 s86, s84, 0x3c90
s_add_u32 s86, s6, s86
s_addc_u32 s87, s7, 0
s_swappc_b64 s[86:87], s[86:87]
s_waitcnt lgkmcnt(0)
v_readfirstlane_b32 s27, v144
v_pk_fma_f16 v24, v29, s82, v24 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v25, v30, s82, v25 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v26, v31, s82, v26 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v27, v32, s82, v27 op_sel:[0,1,0] op_sel_hi:[1,1,1]
s_setprio 0
ds_load_b128 v[44:47], v3 offset:11584
ds_load_b128 v[49:52], v3 offset:12160
s_setprio 2
s_and_not1_b32 null, 0xffffff00, s11
s_cbranch_scc1 25
s_pack_ll_b32_b16 s10, s10, s10
s_bfm_b64 exec, s91, 0
v_cmp_ne_u32 vcc, v21, s90
s_cbranch_vccz 12
buffer_load_b32 v21, off, s[88:91], null glc
s_cmp_eq_u32 s10, 0
s_cselect_b32 vcc_lo, 0, 0x10000
s_add_u32 s10, s10, vcc_lo
s_cbranch_scc1 2
s_waitcnt vmcnt(0)
s_branch 65524
s_and_b32 s91, 0xffff0000, s91
s_mov_b32 s10, 0
s_mov_b64 exec, -1
s_mul_i32 s90, s90, 3
s_and_b32 s90, s90, 0x3f3f3f3f
s_add_u32 s88, s88, 0x100
s_and_b32 s88, s88, 0xfffff7ff
s_bitcmp1_b32 s92, 1
s_cselect_b32 s86, s85, 0x3d54
s_add_u32 s86, s6, s86
s_addc_u32 s87, s7, 0
s_cmp_le_u32 s9, 16
s_cselect_b32 s99, -1, 4
s_sub_u32 s99, s99, 1
s_cselect_b32 s29, s98, s29
s_bitset0_b32 s29, 0
s_swappc_b64 s[86:87], s[86:87]
s_waitcnt lgkmcnt(0)
s_barrier
v_pk_fma_f16 v44, v49, s82, v44 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v45, v50, s82, v45 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v46, v51, s82, v46 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v47, v52, s82, v47 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_mov_b32 v144, s29
v_add_co_u32 v33, vcc, v5, s23
v_pk_mad_u16 v23, v5, 0x20001, s35
v_pk_mad_u16 v28, v5, 0x20001, s38
_v_pk_min_u16__vop3p 22, 289, 261, 0x0, 0x3, 0x0, 0x0
v_cndmask_b32 v43, 0, s42, vcc
v_cndmask_b32 v167, 0, s50, vcc
v_mad_u32_u16 v7, v23, 1, v6 op_sel:[0,0,0,0]
v_mad_u32_u16 v12, v28, 1, v11 op_sel:[0,0,0,0]
v_add3_u32 v6, v6, s36, v43
v_add3_u32 v11, v11, s39, v167
_v_pk_sub_u16__vop3p 22, 261, 278, 0x0, 0x3, 0x0, 0x0
v_add_co_ci_u32 v4, s[54:55], v4, s15, vcc
v_cndmask_b32 v6, v6, 0x80000000, s[54:55]
v_cndmask_b32 v11, v11, 0x80000000, s[54:55]
v_cmp_lt_u16 vcc, v23, s34
v_cndmask_b32 v7, 0x80000000, v7, vcc
v_cmp_lt_u16 vcc, v28, s37
v_cndmask_b32 v12, 0x80000000, v12, vcc
_v_pk_ashrrev_i16__vop3p 22, 143, 278, 0x0, 0x2, 0x0, 0x0
_v_pk_add_u16__vop3p 53, 279, 41, 0x1, 0x3, 0x0, 0x0
_v_pk_add_u16__vop3p 48, 279, 40, 0x1, 0x3, 0x0, 0x0
v_mad_u32_u16 v10, v53, s44, v7 op_sel:[1,0,0,0]
v_mad_u32_u16 v8, v48, s44, v7 op_sel:[1,0,0,0]
_v_pk_add_u16__vop3p 38, 284, 45, 0x1, 0x3, 0x0, 0x0
_v_cmp_lt_u16__vop3 106, 53, 34, 0x3
v_cndmask_b32 v10, 0x80000000, v10, vcc
_v_cmp_lt_u16__vop3 106, 48, 34, 0x3
v_cndmask_b32 v8, 0x80000000, v8, vcc
v_mad_u32_u16 v13, v38, s52, v12 op_sel:[1,0,0,0]
v_mad_u32_u16 v9, v53, s44, v7 op_sel:[0,0,0,0]
v_mad_u32_u16 v7, v48, s44, v7 op_sel:[0,0,0,0]
_v_cmp_lt_u16__vop3 106, 38, 37, 0x3
v_cndmask_b32 v13, 0x80000000, v13, vcc
_v_cmp_lt_u16__vop3 106, 53, 34, 0x2
v_cndmask_b32 v9, 0x80000000, v9, vcc
_v_cmp_lt_u16__vop3 106, 48, 34, 0x2
v_cndmask_b32 v7, 0x80000000, v7, vcc
v_mad_u32_u16 v12, v38, s52, v12 op_sel:[0,0,0,0]
v_pk_mad_u16 v5, v22, s22, v33
_v_cmp_lt_u16__vop3 106, 38, 37, 0x2
v_cndmask_b32 v12, 0x80000000, v12, vcc
v_add_co_u32 v22, vcc, v4, s8
v_cndmask_b32 v144, s98, v144, vcc
s_setprio 0
ds_load_b128 v[34:37], v3
ds_store_b128 v16, v[7:10] offset:37120
ds_load_b128 v[39:42], v3 offset:576
ds_store_b32 v17, v144 offset:39168
s_setprio 2
s_sub_u32 s26, s26, 1
s_cselect_b32 s91, 0x21010000, s91
s_bitcmp1_b32 s92, 2
s_cselect_b32 s86, s84, 0x3c90
s_add_u32 s86, s6, s86
s_addc_u32 s87, s7, 0
s_swappc_b64 s[86:87], s[86:87]
s_waitcnt lgkmcnt(0)
v_add_nc_u32 v15, s53, v14
v_mov_b32 v165, v163
v_mov_b32 v166, v164
v_pk_fma_f16 v147, v34, s82, v24 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v148, v35, s82, v25 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v149, v36, s82, v26 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v150, v37, s82, v27 op_sel:[0,1,0] op_sel_hi:[1,1,1]
_v_pk_add_f16__vop3p 34, 285, 290, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 35, 286, 291, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 36, 287, 292, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 37, 288, 293, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 151, 290, 295, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 152, 291, 296, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 153, 292, 297, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 154, 293, 298, 0x0, 0x3, 0x0, 0x0
s_setprio 0
ds_load_b64 v[163:164], v15 offset:39680
ds_load_b128 v[54:57], v3 offset:2304
ds_load_b128 v[59:62], v3 offset:2880
s_setprio 2
s_mov_b32 s92, s93
s_mov_b32 s93, s94
s_mov_b32 s94, s95
s_mov_b32 s95, s96
s_mov_b32 s96, s97
s_mov_b32 s97, s27
s_bitcmp1_b32 s92, 0
s_cbranch_scc1 2181
s_sub_u32 s49, s49, 1
s_cselect_b32 s49, 3, s49
s_lshl_b32 s53, s49, 9
s_bitcmp1_b32 s92, 1
s_cselect_b32 s86, s85, 0x3c94
s_add_u32 s86, s6, s86
s_addc_u32 s87, s7, 0
s_bitcmp1_b32 s92, 2
s_cselect_b32 s75, 0x11014000, 0
s_sub_u32 s69, s12, 1
s_cselect_b32 s75, 0, s75
s_mov_b64 s[72:73], s[20:21]
s_swappc_b64 s[86:87], s[86:87]
s_waitcnt lgkmcnt(0)
s_barrier
v_pk_fma_f16 v155, v54, s82, v44 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v156, v55, s82, v45 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v157, v56, s82, v46 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v158, v57, s82, v47 op_sel:[0,1,0] op_sel_hi:[1,1,1]
_v_pk_add_f16__vop3p 54, 305, 310, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 55, 306, 311, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 56, 307, 312, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 57, 308, 313, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 159, 310, 315, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 160, 311, 316, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 161, 312, 317, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 162, 313, 318, 0x0, 0x3, 0x0, 0x0
s_add_u32 s11, s11, 0x100
s_cbranch_scc0 7
s_bitset0_b32 s91, 23
s_lshl_b64 exec, 1, s90
buffer_store_b8 v0, off, s[88:91], s4
s_mov_b64 exec, -1
s_mul_i32 s11, s11, 0xffffff01
s_and_not1_b32 null, 0xffffff00, s11
s_cbranch_scc1 3
s_bitset1_b32 s91, 23
buffer_load_b32 v21, off, s[88:91], null glc
s_setprio 0
s_nop 1
ds_load_b128 v[24:27], v3 offset:9280
ds_store_b64 v15, v[12:13] offset:39680
ds_load_b128 v[29:32], v3 offset:9856
ds_load_b32 v144, v18 offset:39168
s_setprio 2
s_bitcmp1_b32 s92, 2
s_cselect_b32 s86, s84, 0x3c90
s_add_u32 s86, s6, s86
s_addc_u32 s87, s7, 0
s_swappc_b64 s[86:87], s[86:87]
s_waitcnt lgkmcnt(0)
v_readfirstlane_b32 s27, v144
v_pk_fma_f16 v24, v29, s82, v24 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v25, v30, s82, v25 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v26, v31, s82, v26 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v27, v32, s82, v27 op_sel:[0,1,0] op_sel_hi:[1,1,1]
s_setprio 0
ds_load_b128 v[44:47], v3 offset:11584
ds_load_b128 v[49:52], v3 offset:12160
s_setprio 2
s_and_not1_b32 null, 0xffffff00, s11
s_cbranch_scc1 25
s_pack_ll_b32_b16 s10, s10, s10
s_bfm_b64 exec, s91, 0
v_cmp_ne_u32 vcc, v21, s90
s_cbranch_vccz 12
buffer_load_b32 v21, off, s[88:91], null glc
s_cmp_eq_u32 s10, 0
s_cselect_b32 vcc_lo, 0, 0x10000
s_add_u32 s10, s10, vcc_lo
s_cbranch_scc1 2
s_waitcnt vmcnt(0)
s_branch 65524
s_and_b32 s91, 0xffff0000, s91
s_mov_b32 s10, 0
s_mov_b64 exec, -1
s_mul_i32 s90, s90, 3
s_and_b32 s90, s90, 0x3f3f3f3f
s_add_u32 s88, s88, 0x100
s_and_b32 s88, s88, 0xfffff7ff
s_bitcmp1_b32 s92, 1
s_cselect_b32 s86, s85, 0x3d54
s_add_u32 s86, s6, s86
s_addc_u32 s87, s7, 0
s_bitcmp1_b32 s27, 1
s_cbranch_scc1 65244
s_branch 65016
s_setpc_b64 s[86:87]
s_bitcmp1_b32 s92, 3
s_cbranch_scc0 40
v_mov_b32 v64, 0
v_mov_b32 v68, 0
v_mov_b32 v65, 0
v_mov_b32 v69, 0
v_mov_b32 v66, 0
v_mov_b32 v70, 0
v_mov_b32 v67, 0
v_mov_b32 v71, 0
v_mov_b32 v80, 0
v_mov_b32 v84, 0
v_mov_b32 v81, 0
v_mov_b32 v85, 0
v_mov_b32 v82, 0
v_mov_b32 v86, 0
v_mov_b32 v83, 0
v_mov_b32 v87, 0
v_mov_b32 v96, 0
v_mov_b32 v100, 0
v_mov_b32 v97, 0
v_mov_b32 v101, 0
v_mov_b32 v98, 0
v_mov_b32 v102, 0
v_mov_b32 v99, 0
v_mov_b32 v103, 0
v_mov_b32 v112, 0
v_mov_b32 v116, 0
v_mov_b32 v113, 0
v_mov_b32 v117, 0
v_mov_b32 v114, 0
v_mov_b32 v118, 0
v_mov_b32 v115, 0
v_mov_b32 v119, 0
v_mov_b32 v128, 0
v_mov_b32 v132, 0
v_mov_b32 v129, 0
v_mov_b32 v133, 0
v_mov_b32 v130, 0
v_mov_b32 v134, 0
v_mov_b32 v131, 0
v_mov_b32 v135, 0
s_mov_b32 s85, 0x3e14
s_cmp_le_u32 s9, 16
s_cmov_b32 s85, 0x3c90
s_setpc_b64 s[86:87]
s_bitcmp1_b32 s92, 3
s_cbranch_scc0 40
v_mov_b32 v72, 0
v_mov_b32 v76, 0
v_mov_b32 v73, 0
v_mov_b32 v77, 0
v_mov_b32 v74, 0
v_mov_b32 v78, 0
v_mov_b32 v75, 0
v_mov_b32 v79, 0
v_mov_b32 v88, 0
v_mov_b32 v92, 0
v_mov_b32 v89, 0
v_mov_b32 v93, 0
v_mov_b32 v90, 0
v_mov_b32 v94, 0
v_mov_b32 v91, 0
v_mov_b32 v95, 0
v_mov_b32 v104, 0
v_mov_b32 v108, 0
v_mov_b32 v105, 0
v_mov_b32 v109, 0
v_mov_b32 v106, 0
v_mov_b32 v110, 0
v_mov_b32 v107, 0
v_mov_b32 v111, 0
v_mov_b32 v120, 0
v_mov_b32 v124, 0
v_mov_b32 v121, 0
v_mov_b32 v125, 0
v_mov_b32 v122, 0
v_mov_b32 v126, 0
v_mov_b32 v123, 0
v_mov_b32 v127, 0
v_mov_b32 v136, 0
v_mov_b32 v140, 0
v_mov_b32 v137, 0
v_mov_b32 v141, 0
v_mov_b32 v138, 0
v_mov_b32 v142, 0
v_mov_b32 v139, 0
v_mov_b32 v143, 0
s_mov_b32 s85, 0x3e14
s_cmp_le_u32 s9, 16
s_cmov_b32 s85, 0x3c90
s_setpc_b64 s[86:87]
_v_pk_add_f16__vop3p 147, 403, 320, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 148, 404, 321, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 149, 405, 322, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 150, 406, 323, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 151, 407, 324, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 152, 408, 325, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 153, 409, 326, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 154, 410, 327, 0x0, 0x3, 0x0, 0x0
v_mov_b32 v64, v147
v_mov_b32 v65, v148
v_mov_b32 v66, v149
v_mov_b32 v67, v150
v_mov_b32 v68, v151
v_mov_b32 v69, v152
v_mov_b32 v70, v153
v_mov_b32 v71, v154
s_mov_b32 s85, 0x3e80
s_setpc_b64 s[86:87]
_v_pk_add_f16__vop3p 155, 411, 328, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 156, 412, 329, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 157, 413, 330, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 158, 414, 331, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 159, 415, 332, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 160, 416, 333, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 161, 417, 334, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 162, 418, 335, 0x0, 0x3, 0x0, 0x0
v_mov_b32 v72, v155
v_mov_b32 v73, v156
v_mov_b32 v74, v157
v_mov_b32 v75, v158
v_mov_b32 v76, v159
v_mov_b32 v77, v160
v_mov_b32 v78, v161
v_mov_b32 v79, v162
s_mov_b32 s85, 0x3eec
s_setpc_b64 s[86:87]
_v_pk_add_f16__vop3p 147, 403, 336, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 148, 404, 337, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 149, 405, 338, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 150, 406, 339, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 151, 407, 340, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 152, 408, 341, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 153, 409, 342, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 154, 410, 343, 0x0, 0x3, 0x0, 0x0
v_mov_b32 v80, v147
v_mov_b32 v81, v148
v_mov_b32 v82, v149
v_mov_b32 v83, v150
v_mov_b32 v84, v151
v_mov_b32 v85, v152
v_mov_b32 v86, v153
v_mov_b32 v87, v154
s_mov_b32 s85, 0x3f58
s_setpc_b64 s[86:87]
_v_pk_add_f16__vop3p 155, 411, 344, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 156, 412, 345, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 157, 413, 346, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 158, 414, 347, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 159, 415, 348, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 160, 416, 349, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 161, 417, 350, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 162, 418, 351, 0x0, 0x3, 0x0, 0x0
v_mov_b32 v88, v155
v_mov_b32 v89, v156
v_mov_b32 v90, v157
v_mov_b32 v91, v158
v_mov_b32 v92, v159
v_mov_b32 v93, v160
v_mov_b32 v94, v161
v_mov_b32 v95, v162
s_mov_b32 s85, 0x3fc4
s_setpc_b64 s[86:87]
_v_pk_add_f16__vop3p 147, 403, 352, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 148, 404, 353, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 149, 405, 354, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 150, 406, 355, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 151, 407, 356, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 152, 408, 357, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 153, 409, 358, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 154, 410, 359, 0x0, 0x3, 0x0, 0x0
v_mov_b32 v96, v147
v_mov_b32 v97, v148
v_mov_b32 v98, v149
v_mov_b32 v99, v150
v_mov_b32 v100, v151
v_mov_b32 v101, v152
v_mov_b32 v102, v153
v_mov_b32 v103, v154
s_mov_b32 s85, 0x4030
s_setpc_b64 s[86:87]
_v_pk_add_f16__vop3p 155, 411, 360, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 156, 412, 361, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 157, 413, 362, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 158, 414, 363, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 159, 415, 364, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 160, 416, 365, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 161, 417, 366, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 162, 418, 367, 0x0, 0x3, 0x0, 0x0
v_mov_b32 v104, v155
v_mov_b32 v105, v156
v_mov_b32 v106, v157
v_mov_b32 v107, v158
v_mov_b32 v108, v159
v_mov_b32 v109, v160
v_mov_b32 v110, v161
v_mov_b32 v111, v162
s_mov_b32 s85, 0x409c
s_setpc_b64 s[86:87]
_v_pk_add_f16__vop3p 147, 403, 368, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 148, 404, 369, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 149, 405, 370, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 150, 406, 371, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 151, 407, 372, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 152, 408, 373, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 153, 409, 374, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 154, 410, 375, 0x0, 0x3, 0x0, 0x0
v_mov_b32 v112, v147
v_mov_b32 v113, v148
v_mov_b32 v114, v149
v_mov_b32 v115, v150
v_mov_b32 v116, v151
v_mov_b32 v117, v152
v_mov_b32 v118, v153
v_mov_b32 v119, v154
s_mov_b32 s85, 0x4108
s_setpc_b64 s[86:87]
_v_pk_add_f16__vop3p 155, 411, 376, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 156, 412, 377, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 157, 413, 378, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 158, 414, 379, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 159, 415, 380, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 160, 416, 381, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 161, 417, 382, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 162, 418, 383, 0x0, 0x3, 0x0, 0x0
v_mov_b32 v120, v155
v_mov_b32 v121, v156
v_mov_b32 v122, v157
v_mov_b32 v123, v158
v_mov_b32 v124, v159
v_mov_b32 v125, v160
v_mov_b32 v126, v161
v_mov_b32 v127, v162
s_mov_b32 s85, 0x4174
s_setpc_b64 s[86:87]
_v_pk_add_f16__vop3p 147, 403, 384, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 148, 404, 385, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 149, 405, 386, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 150, 406, 387, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 151, 407, 388, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 152, 408, 389, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 153, 409, 390, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 154, 410, 391, 0x0, 0x3, 0x0, 0x0
v_mov_b32 v128, v147
v_mov_b32 v129, v148
v_mov_b32 v130, v149
v_mov_b32 v131, v150
v_mov_b32 v132, v151
v_mov_b32 v133, v152
v_mov_b32 v134, v153
v_mov_b32 v135, v154
s_mov_b32 s85, 0x41e0
s_setpc_b64 s[86:87]
_v_pk_add_f16__vop3p 155, 411, 392, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 156, 412, 393, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 157, 413, 394, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 158, 414, 395, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 159, 415, 396, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 160, 416, 397, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 161, 417, 398, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 162, 418, 399, 0x0, 0x3, 0x0, 0x0
v_mov_b32 v136, v155
v_mov_b32 v137, v156
v_mov_b32 v138, v157
v_mov_b32 v139, v158
v_mov_b32 v140, v159
v_mov_b32 v141, v160
v_mov_b32 v142, v161
v_mov_b32 v143, v162
s_mov_b32 s85, 0x3e14
s_setpc_b64 s[86:87]
_v_pk_add_f16__vop3p 147, 403, 56, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 148, 404, 57, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 149, 405, 59, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 150, 406, 64, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 151, 407, 56, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 152, 408, 57, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 153, 409, 59, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 154, 410, 64, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 22, 32, 403, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 28, 32, 404, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 38, 32, 405, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 48, 32, 406, 0x0, 0x3, 0x0, 0x0
_v_pk_max_f16__vop3p 147, 403, 278, 0x0, 0x3, 0x0, 0x0
_v_pk_max_f16__vop3p 148, 404, 284, 0x0, 0x3, 0x0, 0x0
_v_pk_max_f16__vop3p 149, 405, 294, 0x0, 0x3, 0x0, 0x0
_v_pk_max_f16__vop3p 150, 406, 304, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 22, 32, 407, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 28, 32, 408, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 38, 32, 409, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 48, 32, 410, 0x0, 0x3, 0x0, 0x0
_v_pk_max_f16__vop3p 151, 407, 278, 0x0, 0x3, 0x0, 0x0
_v_pk_max_f16__vop3p 152, 408, 284, 0x0, 0x3, 0x0, 0x0
_v_pk_max_f16__vop3p 153, 409, 294, 0x0, 0x3, 0x0, 0x0
_v_pk_max_f16__vop3p 154, 410, 304, 0x0, 0x3, 0x0, 0x0
buffer_store_b16 v147, v165, s[72:75], 0 idxen
buffer_store_b16 v151, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_d16_hi_b16 v147, v165, s[72:75], 0 idxen
buffer_store_d16_hi_b16 v151, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_b16 v148, v165, s[72:75], 0 idxen
buffer_store_b16 v152, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_d16_hi_b16 v148, v165, s[72:75], 0 idxen
buffer_store_d16_hi_b16 v152, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_b16 v149, v165, s[72:75], 0 idxen
buffer_store_b16 v153, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_d16_hi_b16 v149, v165, s[72:75], 0 idxen
buffer_store_d16_hi_b16 v153, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_b16 v150, v165, s[72:75], 0 idxen
buffer_store_b16 v154, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_d16_hi_b16 v150, v165, s[72:75], 0 idxen
buffer_store_d16_hi_b16 v154, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
s_mov_b32 s84, 0x4418
s_setpc_b64 s[86:87]
_v_pk_add_f16__vop3p 155, 411, 65, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 156, 412, 66, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 157, 413, 67, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 158, 414, 68, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 159, 415, 65, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 160, 416, 66, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 161, 417, 67, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 162, 418, 68, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 22, 32, 411, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 28, 32, 412, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 38, 32, 413, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 48, 32, 414, 0x0, 0x3, 0x0, 0x0
_v_pk_max_f16__vop3p 155, 411, 278, 0x0, 0x3, 0x0, 0x0
_v_pk_max_f16__vop3p 156, 412, 284, 0x0, 0x3, 0x0, 0x0
_v_pk_max_f16__vop3p 157, 413, 294, 0x0, 0x3, 0x0, 0x0
_v_pk_max_f16__vop3p 158, 414, 304, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 22, 32, 415, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 28, 32, 416, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 38, 32, 417, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 48, 32, 418, 0x0, 0x3, 0x0, 0x0
_v_pk_max_f16__vop3p 159, 415, 278, 0x0, 0x3, 0x0, 0x0
_v_pk_max_f16__vop3p 160, 416, 284, 0x0, 0x3, 0x0, 0x0
_v_pk_max_f16__vop3p 161, 417, 294, 0x0, 0x3, 0x0, 0x0
_v_pk_max_f16__vop3p 162, 418, 304, 0x0, 0x3, 0x0, 0x0
buffer_store_b16 v155, v165, s[72:75], 0 idxen
buffer_store_b16 v159, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_d16_hi_b16 v155, v165, s[72:75], 0 idxen
buffer_store_d16_hi_b16 v159, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_b16 v156, v165, s[72:75], 0 idxen
buffer_store_b16 v160, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_d16_hi_b16 v156, v165, s[72:75], 0 idxen
buffer_store_d16_hi_b16 v160, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_b16 v157, v165, s[72:75], 0 idxen
buffer_store_b16 v161, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_d16_hi_b16 v157, v165, s[72:75], 0 idxen
buffer_store_d16_hi_b16 v161, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_b16 v158, v165, s[72:75], 0 idxen
buffer_store_b16 v162, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_d16_hi_b16 v158, v165, s[72:75], 0 idxen
buffer_store_d16_hi_b16 v162, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
s_mov_b32 s84, 0x424c
s_setpc_b64 s[86:87]
_v_pk_add_f16__vop3p 147, 403, 56, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 148, 404, 57, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 149, 405, 59, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 150, 406, 64, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 151, 407, 56, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 152, 408, 57, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 153, 409, 59, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 154, 410, 64, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 22, 32, 403, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 28, 32, 404, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 38, 32, 405, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 48, 32, 406, 0x0, 0x3, 0x0, 0x0
_v_pk_min_f16__vop3p 147, 403, 278, 0x0, 0x3, 0x0, 0x0
_v_pk_min_f16__vop3p 148, 404, 284, 0x0, 0x3, 0x0, 0x0
_v_pk_min_f16__vop3p 149, 405, 294, 0x0, 0x3, 0x0, 0x0
_v_pk_min_f16__vop3p 150, 406, 304, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 22, 32, 407, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 28, 32, 408, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 38, 32, 409, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 48, 32, 410, 0x0, 0x3, 0x0, 0x0
_v_pk_min_f16__vop3p 151, 407, 278, 0x0, 0x3, 0x0, 0x0
_v_pk_min_f16__vop3p 152, 408, 284, 0x0, 0x3, 0x0, 0x0
_v_pk_min_f16__vop3p 153, 409, 294, 0x0, 0x3, 0x0, 0x0
_v_pk_min_f16__vop3p 154, 410, 304, 0x0, 0x3, 0x0, 0x0
buffer_store_b16 v147, v165, s[72:75], 0 idxen
buffer_store_b16 v151, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_d16_hi_b16 v147, v165, s[72:75], 0 idxen
buffer_store_d16_hi_b16 v151, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_b16 v148, v165, s[72:75], 0 idxen
buffer_store_b16 v152, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_d16_hi_b16 v148, v165, s[72:75], 0 idxen
buffer_store_d16_hi_b16 v152, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_b16 v149, v165, s[72:75], 0 idxen
buffer_store_b16 v153, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_d16_hi_b16 v149, v165, s[72:75], 0 idxen
buffer_store_d16_hi_b16 v153, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_b16 v150, v165, s[72:75], 0 idxen
buffer_store_b16 v154, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_d16_hi_b16 v150, v165, s[72:75], 0 idxen
buffer_store_d16_hi_b16 v154, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
s_mov_b32 s84, 0x47b0
s_setpc_b64 s[86:87]
_v_pk_add_f16__vop3p 155, 411, 65, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 156, 412, 66, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 157, 413, 67, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 158, 414, 68, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 159, 415, 65, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 160, 416, 66, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 161, 417, 67, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 162, 418, 68, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 22, 32, 411, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 28, 32, 412, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 38, 32, 413, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 48, 32, 414, 0x0, 0x3, 0x0, 0x0
_v_pk_min_f16__vop3p 155, 411, 278, 0x0, 0x3, 0x0, 0x0
_v_pk_min_f16__vop3p 156, 412, 284, 0x0, 0x3, 0x0, 0x0
_v_pk_min_f16__vop3p 157, 413, 294, 0x0, 0x3, 0x0, 0x0
_v_pk_min_f16__vop3p 158, 414, 304, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 22, 32, 415, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 28, 32, 416, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 38, 32, 417, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 48, 32, 418, 0x0, 0x3, 0x0, 0x0
_v_pk_min_f16__vop3p 159, 415, 278, 0x0, 0x3, 0x0, 0x0
_v_pk_min_f16__vop3p 160, 416, 284, 0x0, 0x3, 0x0, 0x0
_v_pk_min_f16__vop3p 161, 417, 294, 0x0, 0x3, 0x0, 0x0
_v_pk_min_f16__vop3p 162, 418, 304, 0x0, 0x3, 0x0, 0x0
buffer_store_b16 v155, v165, s[72:75], 0 idxen
buffer_store_b16 v159, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_d16_hi_b16 v155, v165, s[72:75], 0 idxen
buffer_store_d16_hi_b16 v159, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_b16 v156, v165, s[72:75], 0 idxen
buffer_store_b16 v160, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_d16_hi_b16 v156, v165, s[72:75], 0 idxen
buffer_store_d16_hi_b16 v160, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_b16 v157, v165, s[72:75], 0 idxen
buffer_store_b16 v161, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_d16_hi_b16 v157, v165, s[72:75], 0 idxen
buffer_store_d16_hi_b16 v161, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_b16 v158, v165, s[72:75], 0 idxen
buffer_store_b16 v162, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_d16_hi_b16 v158, v165, s[72:75], 0 idxen
buffer_store_d16_hi_b16 v162, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
s_mov_b32 s84, 0x45e4
s_setpc_b64 s[86:87]
_v_pk_add_f16__vop3p 147, 403, 56, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 148, 404, 57, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 149, 405, 59, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 150, 406, 64, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 151, 407, 56, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 152, 408, 57, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 153, 409, 59, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 154, 410, 64, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p_lit 147, 0xbdc5bdc5, 403, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 148, 0xbdc5bdc5, 404, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 149, 0xbdc5bdc5, 405, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 150, 0xbdc5bdc5, 406, 0x0, 0x3
v_exp_f16 v147, v147
v_exp_f16 v148, v148
v_exp_f16 v149, v149
v_exp_f16 v150, v150
_v_exp_f16__vop3 147, 147, 0x9
_v_exp_f16__vop3 148, 148, 0x9
_v_exp_f16__vop3 149, 149, 0x9
_v_exp_f16__vop3 150, 150, 0x9
_v_pk_add_f16__vop3p_lit 147, 0x3c003c00, 403, 0x0, 0x3
_v_pk_add_f16__vop3p_lit 148, 0x3c003c00, 404, 0x0, 0x3
_v_pk_add_f16__vop3p_lit 149, 0x3c003c00, 405, 0x0, 0x3
_v_pk_add_f16__vop3p_lit 150, 0x3c003c00, 406, 0x0, 0x3
v_rcp_f16 v147, v147
v_rcp_f16 v148, v148
v_rcp_f16 v149, v149
v_rcp_f16 v150, v150
_v_rcp_f16__vop3 147, 147, 0x9
_v_rcp_f16__vop3 148, 148, 0x9
_v_rcp_f16__vop3 149, 149, 0x9
_v_rcp_f16__vop3 150, 150, 0x9
_v_pk_mul_f16__vop3p_lit 151, 0xbdc5bdc5, 407, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 152, 0xbdc5bdc5, 408, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 153, 0xbdc5bdc5, 409, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 154, 0xbdc5bdc5, 410, 0x0, 0x3
v_exp_f16 v151, v151
v_exp_f16 v152, v152
v_exp_f16 v153, v153
v_exp_f16 v154, v154
_v_exp_f16__vop3 151, 151, 0x9
_v_exp_f16__vop3 152, 152, 0x9
_v_exp_f16__vop3 153, 153, 0x9
_v_exp_f16__vop3 154, 154, 0x9
_v_pk_add_f16__vop3p_lit 151, 0x3c003c00, 407, 0x0, 0x3
_v_pk_add_f16__vop3p_lit 152, 0x3c003c00, 408, 0x0, 0x3
_v_pk_add_f16__vop3p_lit 153, 0x3c003c00, 409, 0x0, 0x3
_v_pk_add_f16__vop3p_lit 154, 0x3c003c00, 410, 0x0, 0x3
v_rcp_f16 v151, v151
v_rcp_f16 v152, v152
v_rcp_f16 v153, v153
v_rcp_f16 v154, v154
_v_rcp_f16__vop3 151, 151, 0x9
_v_rcp_f16__vop3 152, 152, 0x9
_v_rcp_f16__vop3 153, 153, 0x9
_v_rcp_f16__vop3 154, 154, 0x9
buffer_store_b16 v147, v165, s[72:75], 0 idxen
buffer_store_b16 v151, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_d16_hi_b16 v147, v165, s[72:75], 0 idxen
buffer_store_d16_hi_b16 v151, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_b16 v148, v165, s[72:75], 0 idxen
buffer_store_b16 v152, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_d16_hi_b16 v148, v165, s[72:75], 0 idxen
buffer_store_d16_hi_b16 v152, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_b16 v149, v165, s[72:75], 0 idxen
buffer_store_b16 v153, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_d16_hi_b16 v149, v165, s[72:75], 0 idxen
buffer_store_d16_hi_b16 v153, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_b16 v150, v165, s[72:75], 0 idxen
buffer_store_b16 v154, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_d16_hi_b16 v150, v165, s[72:75], 0 idxen
buffer_store_d16_hi_b16 v154, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
s_mov_b32 s84, 0x4c88
s_setpc_b64 s[86:87]
_v_pk_add_f16__vop3p 155, 411, 65, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 156, 412, 66, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 157, 413, 67, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 158, 414, 68, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 159, 415, 65, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 160, 416, 66, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 161, 417, 67, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 162, 418, 68, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p_lit 155, 0xbdc5bdc5, 411, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 156, 0xbdc5bdc5, 412, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 157, 0xbdc5bdc5, 413, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 158, 0xbdc5bdc5, 414, 0x0, 0x3
v_exp_f16 v155, v155
v_exp_f16 v156, v156
v_exp_f16 v157, v157
v_exp_f16 v158, v158
_v_exp_f16__vop3 155, 155, 0x9
_v_exp_f16__vop3 156, 156, 0x9
_v_exp_f16__vop3 157, 157, 0x9
_v_exp_f16__vop3 158, 158, 0x9
_v_pk_add_f16__vop3p_lit 155, 0x3c003c00, 411, 0x0, 0x3
_v_pk_add_f16__vop3p_lit 156, 0x3c003c00, 412, 0x0, 0x3
_v_pk_add_f16__vop3p_lit 157, 0x3c003c00, 413, 0x0, 0x3
_v_pk_add_f16__vop3p_lit 158, 0x3c003c00, 414, 0x0, 0x3
v_rcp_f16 v155, v155
v_rcp_f16 v156, v156
v_rcp_f16 v157, v157
v_rcp_f16 v158, v158
_v_rcp_f16__vop3 155, 155, 0x9
_v_rcp_f16__vop3 156, 156, 0x9
_v_rcp_f16__vop3 157, 157, 0x9
_v_rcp_f16__vop3 158, 158, 0x9
_v_pk_mul_f16__vop3p_lit 159, 0xbdc5bdc5, 415, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 160, 0xbdc5bdc5, 416, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 161, 0xbdc5bdc5, 417, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 162, 0xbdc5bdc5, 418, 0x0, 0x3
v_exp_f16 v159, v159
v_exp_f16 v160, v160
v_exp_f16 v161, v161
v_exp_f16 v162, v162
_v_exp_f16__vop3 159, 159, 0x9
_v_exp_f16__vop3 160, 160, 0x9
_v_exp_f16__vop3 161, 161, 0x9
_v_exp_f16__vop3 162, 162, 0x9
_v_pk_add_f16__vop3p_lit 159, 0x3c003c00, 415, 0x0, 0x3
_v_pk_add_f16__vop3p_lit 160, 0x3c003c00, 416, 0x0, 0x3
_v_pk_add_f16__vop3p_lit 161, 0x3c003c00, 417, 0x0, 0x3
_v_pk_add_f16__vop3p_lit 162, 0x3c003c00, 418, 0x0, 0x3
v_rcp_f16 v159, v159
v_rcp_f16 v160, v160
v_rcp_f16 v161, v161
v_rcp_f16 v162, v162
_v_rcp_f16__vop3 159, 159, 0x9
_v_rcp_f16__vop3 160, 160, 0x9
_v_rcp_f16__vop3 161, 161, 0x9
_v_rcp_f16__vop3 162, 162, 0x9
buffer_store_b16 v155, v165, s[72:75], 0 idxen
buffer_store_b16 v159, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_d16_hi_b16 v155, v165, s[72:75], 0 idxen
buffer_store_d16_hi_b16 v159, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_b16 v156, v165, s[72:75], 0 idxen
buffer_store_b16 v160, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_d16_hi_b16 v156, v165, s[72:75], 0 idxen
buffer_store_d16_hi_b16 v160, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_b16 v157, v165, s[72:75], 0 idxen
buffer_store_b16 v161, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_d16_hi_b16 v157, v165, s[72:75], 0 idxen
buffer_store_d16_hi_b16 v161, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_b16 v158, v165, s[72:75], 0 idxen
buffer_store_b16 v162, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_d16_hi_b16 v158, v165, s[72:75], 0 idxen
buffer_store_d16_hi_b16 v162, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
s_mov_b32 s84, 0x497c
s_setpc_b64 s[86:87]
_v_pk_add_f16__vop3p 147, 403, 56, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 148, 404, 57, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 149, 405, 59, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 150, 406, 64, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 151, 407, 56, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 152, 408, 57, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 153, 409, 59, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 154, 410, 64, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 147, 403, 33, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 148, 404, 33, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 149, 405, 33, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 150, 406, 33, 0x0, 0x3, 0x0, 0x0
v_and_b32 v22, 0x7fff7fff, v147
v_and_b32 v28, 0x7fff7fff, v148
v_and_b32 v38, 0x7fff7fff, v149
v_and_b32 v48, 0x7fff7fff, v150
v_mov_b32 v23, 0xb5f8b5f8
v_mov_b32 v33, 0xb5f8b5f8
v_mov_b32 v43, 0xb5f8b5f8
v_mov_b32 v53, 0xb5f8b5f8
v_pk_fma_f16 v23, v22, 0x2ff12ff1, v23
v_pk_fma_f16 v33, v28, 0x2ff12ff1, v33
v_pk_fma_f16 v43, v38, 0x2ff12ff1, v43
v_pk_fma_f16 v53, v48, 0x2ff12ff1, v53
v_pk_fma_f16 v23, v22, v23, 0x1c571c57
v_pk_fma_f16 v33, v28, v33, 0x1c571c57
v_pk_fma_f16 v43, v38, v43, 0x1c571c57
v_pk_fma_f16 v53, v48, v53, 0x1c571c57
v_pk_fma_f16 v23, v22, v23, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_pk_fma_f16 v33, v28, v33, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_pk_fma_f16 v43, v38, v43, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_pk_fma_f16 v53, v48, v53, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
_v_pk_mul_f16__vop3p 23, 278, 279, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 33, 284, 289, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 43, 294, 299, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 53, 304, 309, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p_lit 22, 0x41c541c5, 278, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 28, 0x41c541c5, 284, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 38, 0x41c541c5, 294, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 48, 0x41c541c5, 304, 0x0, 0x3
v_exp_f16 v22, v22
v_exp_f16 v28, v28
v_exp_f16 v38, v38
v_exp_f16 v48, v48
_v_exp_f16__vop1 (22 | /*op_sel*/ 0x80), (22 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (28 | /*op_sel*/ 0x80), (28 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (38 | /*op_sel*/ 0x80), (38 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (48 | /*op_sel*/ 0x80), (48 | /*op_sel*/ 0x80)
_v_pk_add_f16__vop3p 22, 242, 278, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 28, 242, 284, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 38, 242, 294, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 48, 242, 304, 0x0, 0x2, 0x0, 0x0
v_rcp_f16 v22, v22
v_rcp_f16 v28, v28
v_rcp_f16 v38, v38
v_rcp_f16 v48, v48
_v_rcp_f16__vop1 (22 | /*op_sel*/ 0x80), (22 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (28 | /*op_sel*/ 0x80), (28 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (38 | /*op_sel*/ 0x80), (38 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (48 | /*op_sel*/ 0x80), (48 | /*op_sel*/ 0x80)
v_pk_fma_f16 v22, v22, 2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0] neg_lo:[1,0,0] neg_hi:[1,0,0]
v_pk_fma_f16 v28, v28, 2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0] neg_lo:[1,0,0] neg_hi:[1,0,0]
v_pk_fma_f16 v38, v38, 2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0] neg_lo:[1,0,0] neg_hi:[1,0,0]
v_pk_fma_f16 v48, v48, 2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0] neg_lo:[1,0,0] neg_hi:[1,0,0]
_v_cmp_gt_f16__vop3_v_lit 106, 147, 0x38b838b8, 0x0, 0x1
v_cndmask_b16 v23, v23, v22, vcc
_v_cmp_gt_f16__vop3_v_lit 106, 148, 0x38b838b8, 0x0, 0x1
v_cndmask_b16 v33, v33, v28, vcc
_v_cmp_gt_f16__vop3_v_lit 106, 149, 0x38b838b8, 0x0, 0x1
v_cndmask_b16 v43, v43, v38, vcc
_v_cmp_gt_f16__vop3_v_lit 106, 150, 0x38b838b8, 0x0, 0x1
v_cndmask_b16 v53, v53, v48, vcc
_v_cmp_gt_f16__vop3_v_lit 106, 147, 0x38b838b8, 0x3, 0x1
_v_cndmask_b16__vop3 23, 23, 22, 106, 0xb
_v_cmp_gt_f16__vop3_v_lit 106, 148, 0x38b838b8, 0x3, 0x1
_v_cndmask_b16__vop3 33, 33, 28, 106, 0xb
_v_cmp_gt_f16__vop3_v_lit 106, 149, 0x38b838b8, 0x3, 0x1
_v_cndmask_b16__vop3 43, 43, 38, 106, 0xb
_v_cmp_gt_f16__vop3_v_lit 106, 150, 0x38b838b8, 0x3, 0x1
_v_cndmask_b16__vop3 53, 53, 48, 106, 0xb
v_bfi_b32 v147, 0x7fff7fff, v23, v147
v_bfi_b32 v148, 0x7fff7fff, v33, v148
v_bfi_b32 v149, 0x7fff7fff, v43, v149
v_bfi_b32 v150, 0x7fff7fff, v53, v150
_v_pk_mul_f16__vop3p 147, 403, 32, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 148, 404, 32, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 149, 405, 32, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 150, 406, 32, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 151, 407, 33, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 152, 408, 33, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 153, 409, 33, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 154, 410, 33, 0x0, 0x3, 0x0, 0x0
v_and_b32 v22, 0x7fff7fff, v151
v_and_b32 v28, 0x7fff7fff, v152
v_and_b32 v38, 0x7fff7fff, v153
v_and_b32 v48, 0x7fff7fff, v154
v_mov_b32 v23, 0xb5f8b5f8
v_mov_b32 v33, 0xb5f8b5f8
v_mov_b32 v43, 0xb5f8b5f8
v_mov_b32 v53, 0xb5f8b5f8
v_pk_fma_f16 v23, v22, 0x2ff12ff1, v23
v_pk_fma_f16 v33, v28, 0x2ff12ff1, v33
v_pk_fma_f16 v43, v38, 0x2ff12ff1, v43
v_pk_fma_f16 v53, v48, 0x2ff12ff1, v53
v_pk_fma_f16 v23, v22, v23, 0x1c571c57
v_pk_fma_f16 v33, v28, v33, 0x1c571c57
v_pk_fma_f16 v43, v38, v43, 0x1c571c57
v_pk_fma_f16 v53, v48, v53, 0x1c571c57
v_pk_fma_f16 v23, v22, v23, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_pk_fma_f16 v33, v28, v33, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_pk_fma_f16 v43, v38, v43, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_pk_fma_f16 v53, v48, v53, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
_v_pk_mul_f16__vop3p 23, 278, 279, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 33, 284, 289, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 43, 294, 299, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 53, 304, 309, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p_lit 22, 0x41c541c5, 278, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 28, 0x41c541c5, 284, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 38, 0x41c541c5, 294, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 48, 0x41c541c5, 304, 0x0, 0x3
v_exp_f16 v22, v22
v_exp_f16 v28, v28
v_exp_f16 v38, v38
v_exp_f16 v48, v48
_v_exp_f16__vop1 (22 | /*op_sel*/ 0x80), (22 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (28 | /*op_sel*/ 0x80), (28 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (38 | /*op_sel*/ 0x80), (38 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (48 | /*op_sel*/ 0x80), (48 | /*op_sel*/ 0x80)
_v_pk_add_f16__vop3p 22, 242, 278, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 28, 242, 284, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 38, 242, 294, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 48, 242, 304, 0x0, 0x2, 0x0, 0x0
v_rcp_f16 v22, v22
v_rcp_f16 v28, v28
v_rcp_f16 v38, v38
v_rcp_f16 v48, v48
_v_rcp_f16__vop1 (22 | /*op_sel*/ 0x80), (22 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (28 | /*op_sel*/ 0x80), (28 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (38 | /*op_sel*/ 0x80), (38 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (48 | /*op_sel*/ 0x80), (48 | /*op_sel*/ 0x80)
v_pk_fma_f16 v22, v22, 2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0] neg_lo:[1,0,0] neg_hi:[1,0,0]
v_pk_fma_f16 v28, v28, 2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0] neg_lo:[1,0,0] neg_hi:[1,0,0]
v_pk_fma_f16 v38, v38, 2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0] neg_lo:[1,0,0] neg_hi:[1,0,0]
v_pk_fma_f16 v48, v48, 2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0] neg_lo:[1,0,0] neg_hi:[1,0,0]
_v_cmp_gt_f16__vop3_v_lit 106, 151, 0x38b838b8, 0x0, 0x1
v_cndmask_b16 v23, v23, v22, vcc
_v_cmp_gt_f16__vop3_v_lit 106, 152, 0x38b838b8, 0x0, 0x1
v_cndmask_b16 v33, v33, v28, vcc
_v_cmp_gt_f16__vop3_v_lit 106, 153, 0x38b838b8, 0x0, 0x1
v_cndmask_b16 v43, v43, v38, vcc
_v_cmp_gt_f16__vop3_v_lit 106, 154, 0x38b838b8, 0x0, 0x1
v_cndmask_b16 v53, v53, v48, vcc
_v_cmp_gt_f16__vop3_v_lit 106, 151, 0x38b838b8, 0x3, 0x1
_v_cndmask_b16__vop3 23, 23, 22, 106, 0xb
_v_cmp_gt_f16__vop3_v_lit 106, 152, 0x38b838b8, 0x3, 0x1
_v_cndmask_b16__vop3 33, 33, 28, 106, 0xb
_v_cmp_gt_f16__vop3_v_lit 106, 153, 0x38b838b8, 0x3, 0x1
_v_cndmask_b16__vop3 43, 43, 38, 106, 0xb
_v_cmp_gt_f16__vop3_v_lit 106, 154, 0x38b838b8, 0x3, 0x1
_v_cndmask_b16__vop3 53, 53, 48, 106, 0xb
v_bfi_b32 v151, 0x7fff7fff, v23, v151
v_bfi_b32 v152, 0x7fff7fff, v33, v152
v_bfi_b32 v153, 0x7fff7fff, v43, v153
v_bfi_b32 v154, 0x7fff7fff, v53, v154
_v_pk_mul_f16__vop3p 151, 407, 32, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 152, 408, 32, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 153, 409, 32, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 154, 410, 32, 0x0, 0x3, 0x0, 0x0
buffer_store_b16 v147, v165, s[72:75], 0 idxen
buffer_store_b16 v151, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_d16_hi_b16 v147, v165, s[72:75], 0 idxen
buffer_store_d16_hi_b16 v151, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_b16 v148, v165, s[72:75], 0 idxen
buffer_store_b16 v152, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_d16_hi_b16 v148, v165, s[72:75], 0 idxen
buffer_store_d16_hi_b16 v152, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_b16 v149, v165, s[72:75], 0 idxen
buffer_store_b16 v153, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_d16_hi_b16 v149, v165, s[72:75], 0 idxen
buffer_store_d16_hi_b16 v153, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_b16 v150, v165, s[72:75], 0 idxen
buffer_store_b16 v154, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_d16_hi_b16 v150, v165, s[72:75], 0 idxen
buffer_store_d16_hi_b16 v154, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
s_mov_b32 s84, 0x5620
s_setpc_b64 s[86:87]
_v_pk_add_f16__vop3p 155, 411, 65, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 156, 412, 66, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 157, 413, 67, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 158, 414, 68, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 159, 415, 65, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 160, 416, 66, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 161, 417, 67, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 162, 418, 68, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 155, 411, 33, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 156, 412, 33, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 157, 413, 33, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 158, 414, 33, 0x0, 0x3, 0x0, 0x0
v_and_b32 v22, 0x7fff7fff, v155
v_and_b32 v28, 0x7fff7fff, v156
v_and_b32 v38, 0x7fff7fff, v157
v_and_b32 v48, 0x7fff7fff, v158
v_mov_b32 v23, 0xb5f8b5f8
v_mov_b32 v33, 0xb5f8b5f8
v_mov_b32 v43, 0xb5f8b5f8
v_mov_b32 v53, 0xb5f8b5f8
v_pk_fma_f16 v23, v22, 0x2ff12ff1, v23
v_pk_fma_f16 v33, v28, 0x2ff12ff1, v33
v_pk_fma_f16 v43, v38, 0x2ff12ff1, v43
v_pk_fma_f16 v53, v48, 0x2ff12ff1, v53
v_pk_fma_f16 v23, v22, v23, 0x1c571c57
v_pk_fma_f16 v33, v28, v33, 0x1c571c57
v_pk_fma_f16 v43, v38, v43, 0x1c571c57
v_pk_fma_f16 v53, v48, v53, 0x1c571c57
v_pk_fma_f16 v23, v22, v23, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_pk_fma_f16 v33, v28, v33, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_pk_fma_f16 v43, v38, v43, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_pk_fma_f16 v53, v48, v53, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
_v_pk_mul_f16__vop3p 23, 278, 279, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 33, 284, 289, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 43, 294, 299, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 53, 304, 309, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p_lit 22, 0x41c541c5, 278, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 28, 0x41c541c5, 284, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 38, 0x41c541c5, 294, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 48, 0x41c541c5, 304, 0x0, 0x3
v_exp_f16 v22, v22
v_exp_f16 v28, v28
v_exp_f16 v38, v38
v_exp_f16 v48, v48
_v_exp_f16__vop1 (22 | /*op_sel*/ 0x80), (22 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (28 | /*op_sel*/ 0x80), (28 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (38 | /*op_sel*/ 0x80), (38 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (48 | /*op_sel*/ 0x80), (48 | /*op_sel*/ 0x80)
_v_pk_add_f16__vop3p 22, 242, 278, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 28, 242, 284, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 38, 242, 294, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 48, 242, 304, 0x0, 0x2, 0x0, 0x0
v_rcp_f16 v22, v22
v_rcp_f16 v28, v28
v_rcp_f16 v38, v38
v_rcp_f16 v48, v48
_v_rcp_f16__vop1 (22 | /*op_sel*/ 0x80), (22 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (28 | /*op_sel*/ 0x80), (28 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (38 | /*op_sel*/ 0x80), (38 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (48 | /*op_sel*/ 0x80), (48 | /*op_sel*/ 0x80)
v_pk_fma_f16 v22, v22, 2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0] neg_lo:[1,0,0] neg_hi:[1,0,0]
v_pk_fma_f16 v28, v28, 2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0] neg_lo:[1,0,0] neg_hi:[1,0,0]
v_pk_fma_f16 v38, v38, 2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0] neg_lo:[1,0,0] neg_hi:[1,0,0]
v_pk_fma_f16 v48, v48, 2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0] neg_lo:[1,0,0] neg_hi:[1,0,0]
_v_cmp_gt_f16__vop3_v_lit 106, 155, 0x38b838b8, 0x0, 0x1
v_cndmask_b16 v23, v23, v22, vcc
_v_cmp_gt_f16__vop3_v_lit 106, 156, 0x38b838b8, 0x0, 0x1
v_cndmask_b16 v33, v33, v28, vcc
_v_cmp_gt_f16__vop3_v_lit 106, 157, 0x38b838b8, 0x0, 0x1
v_cndmask_b16 v43, v43, v38, vcc
_v_cmp_gt_f16__vop3_v_lit 106, 158, 0x38b838b8, 0x0, 0x1
v_cndmask_b16 v53, v53, v48, vcc
_v_cmp_gt_f16__vop3_v_lit 106, 155, 0x38b838b8, 0x3, 0x1
_v_cndmask_b16__vop3 23, 23, 22, 106, 0xb
_v_cmp_gt_f16__vop3_v_lit 106, 156, 0x38b838b8, 0x3, 0x1
_v_cndmask_b16__vop3 33, 33, 28, 106, 0xb
_v_cmp_gt_f16__vop3_v_lit 106, 157, 0x38b838b8, 0x3, 0x1
_v_cndmask_b16__vop3 43, 43, 38, 106, 0xb
_v_cmp_gt_f16__vop3_v_lit 106, 158, 0x38b838b8, 0x3, 0x1
_v_cndmask_b16__vop3 53, 53, 48, 106, 0xb
v_bfi_b32 v155, 0x7fff7fff, v23, v155
v_bfi_b32 v156, 0x7fff7fff, v33, v156
v_bfi_b32 v157, 0x7fff7fff, v43, v157
v_bfi_b32 v158, 0x7fff7fff, v53, v158
_v_pk_mul_f16__vop3p 155, 411, 32, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 156, 412, 32, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 157, 413, 32, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 158, 414, 32, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 159, 415, 33, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 160, 416, 33, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 161, 417, 33, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 162, 418, 33, 0x0, 0x3, 0x0, 0x0
v_and_b32 v22, 0x7fff7fff, v159
v_and_b32 v28, 0x7fff7fff, v160
v_and_b32 v38, 0x7fff7fff, v161
v_and_b32 v48, 0x7fff7fff, v162
v_mov_b32 v23, 0xb5f8b5f8
v_mov_b32 v33, 0xb5f8b5f8
v_mov_b32 v43, 0xb5f8b5f8
v_mov_b32 v53, 0xb5f8b5f8
v_pk_fma_f16 v23, v22, 0x2ff12ff1, v23
v_pk_fma_f16 v33, v28, 0x2ff12ff1, v33
v_pk_fma_f16 v43, v38, 0x2ff12ff1, v43
v_pk_fma_f16 v53, v48, 0x2ff12ff1, v53
v_pk_fma_f16 v23, v22, v23, 0x1c571c57
v_pk_fma_f16 v33, v28, v33, 0x1c571c57
v_pk_fma_f16 v43, v38, v43, 0x1c571c57
v_pk_fma_f16 v53, v48, v53, 0x1c571c57
v_pk_fma_f16 v23, v22, v23, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_pk_fma_f16 v33, v28, v33, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_pk_fma_f16 v43, v38, v43, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_pk_fma_f16 v53, v48, v53, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
_v_pk_mul_f16__vop3p 23, 278, 279, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 33, 284, 289, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 43, 294, 299, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 53, 304, 309, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p_lit 22, 0x41c541c5, 278, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 28, 0x41c541c5, 284, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 38, 0x41c541c5, 294, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 48, 0x41c541c5, 304, 0x0, 0x3
v_exp_f16 v22, v22
v_exp_f16 v28, v28
v_exp_f16 v38, v38
v_exp_f16 v48, v48
_v_exp_f16__vop1 (22 | /*op_sel*/ 0x80), (22 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (28 | /*op_sel*/ 0x80), (28 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (38 | /*op_sel*/ 0x80), (38 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (48 | /*op_sel*/ 0x80), (48 | /*op_sel*/ 0x80)
_v_pk_add_f16__vop3p 22, 242, 278, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 28, 242, 284, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 38, 242, 294, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 48, 242, 304, 0x0, 0x2, 0x0, 0x0
v_rcp_f16 v22, v22
v_rcp_f16 v28, v28
v_rcp_f16 v38, v38
v_rcp_f16 v48, v48
_v_rcp_f16__vop1 (22 | /*op_sel*/ 0x80), (22 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (28 | /*op_sel*/ 0x80), (28 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (38 | /*op_sel*/ 0x80), (38 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (48 | /*op_sel*/ 0x80), (48 | /*op_sel*/ 0x80)
v_pk_fma_f16 v22, v22, 2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0] neg_lo:[1,0,0] neg_hi:[1,0,0]
v_pk_fma_f16 v28, v28, 2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0] neg_lo:[1,0,0] neg_hi:[1,0,0]
v_pk_fma_f16 v38, v38, 2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0] neg_lo:[1,0,0] neg_hi:[1,0,0]
v_pk_fma_f16 v48, v48, 2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0] neg_lo:[1,0,0] neg_hi:[1,0,0]
_v_cmp_gt_f16__vop3_v_lit 106, 159, 0x38b838b8, 0x0, 0x1
v_cndmask_b16 v23, v23, v22, vcc
_v_cmp_gt_f16__vop3_v_lit 106, 160, 0x38b838b8, 0x0, 0x1
v_cndmask_b16 v33, v33, v28, vcc
_v_cmp_gt_f16__vop3_v_lit 106, 161, 0x38b838b8, 0x0, 0x1
v_cndmask_b16 v43, v43, v38, vcc
_v_cmp_gt_f16__vop3_v_lit 106, 162, 0x38b838b8, 0x0, 0x1
v_cndmask_b16 v53, v53, v48, vcc
_v_cmp_gt_f16__vop3_v_lit 106, 159, 0x38b838b8, 0x3, 0x1
_v_cndmask_b16__vop3 23, 23, 22, 106, 0xb
_v_cmp_gt_f16__vop3_v_lit 106, 160, 0x38b838b8, 0x3, 0x1
_v_cndmask_b16__vop3 33, 33, 28, 106, 0xb
_v_cmp_gt_f16__vop3_v_lit 106, 161, 0x38b838b8, 0x3, 0x1
_v_cndmask_b16__vop3 43, 43, 38, 106, 0xb
_v_cmp_gt_f16__vop3_v_lit 106, 162, 0x38b838b8, 0x3, 0x1
_v_cndmask_b16__vop3 53, 53, 48, 106, 0xb
v_bfi_b32 v159, 0x7fff7fff, v23, v159
v_bfi_b32 v160, 0x7fff7fff, v33, v160
v_bfi_b32 v161, 0x7fff7fff, v43, v161
v_bfi_b32 v162, 0x7fff7fff, v53, v162
_v_pk_mul_f16__vop3p 159, 415, 32, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 160, 416, 32, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 161, 417, 32, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 162, 418, 32, 0x0, 0x3, 0x0, 0x0
buffer_store_b16 v155, v165, s[72:75], 0 idxen
buffer_store_b16 v159, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_d16_hi_b16 v155, v165, s[72:75], 0 idxen
buffer_store_d16_hi_b16 v159, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_b16 v156, v165, s[72:75], 0 idxen
buffer_store_b16 v160, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_d16_hi_b16 v156, v165, s[72:75], 0 idxen
buffer_store_d16_hi_b16 v160, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_b16 v157, v165, s[72:75], 0 idxen
buffer_store_b16 v161, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_d16_hi_b16 v157, v165, s[72:75], 0 idxen
buffer_store_d16_hi_b16 v161, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_b16 v158, v165, s[72:75], 0 idxen
buffer_store_b16 v162, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
buffer_store_d16_hi_b16 v158, v165, s[72:75], 0 idxen
buffer_store_d16_hi_b16 v162, v166, s[72:75], 0 idxen
s_sub_u32 s69, s69, 1
s_cselect_b32 s75, 0, s75
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
s_mov_b32 s84, 0x4f94
s_setpc_b64 s[86:87]
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

