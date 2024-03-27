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
s_mov_b32 s10, 0x5c88
s_mov_b32 s11, 0x31014000
buffer_load_b32 v2, v1, s[8:11], 0 offen
s_waitcnt vmcnt(0)
s_getpc_b64 s[6:7]
s_load_b512 s[8:23], s[2:3], null
s_load_b512 s[24:39], s[2:3], 0x40
s_load_b512 s[40:55], s[2:3], 0x80
s_load_b128 s[56:59], s[2:3], 0xc0
s_load_b64 s[60:61], s[2:3], 0xd0
v_and_b32 v9, 0xff, v0
v_lshrrev_b32 v10, 1, v9
v_and_b32 v11, 1, v0
v_add_nc_u32 v6, v10, 32
v_bfi_b32 v2, 31, v9, v10
v_bfe_u32 v4, v9, 5, 1
v_bfi_b32 v2, 0xbf, v2, v6
v_and_b32 v8, 0x60, v10
v_lshlrev_b32 v2, 4, v2
v_add_lshl_u32 v3, v10, v8, 4
v_mad_u32_u24 v2, v4, 0x800, v2
v_mad_u32_u16 v3, 0x1040, v11, v3 op_sel:[0,0,0,0]
v_xor_b32 v1, 0x600, v2
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
s_cbranch_scc1 5801
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
v_mad_u32_u16 v14, v16, 0x1040, v17 op_sel:[0,0,0,0]
v_bfe_u32 v16, v20, 0, 2
v_mad_u32_u16 v14, v16, 0x80, v14 op_sel:[0,0,0,0]
v_bfe_u32 v17, v20, 4, 2
v_mad_u32_u16 v14, v17, 32, v14 op_sel:[0,0,0,0]
v_bfe_u32 v16, v20, 6, 1
v_mad_u32_u16 v14, v16, 0x400, v14 op_sel:[0,0,0,0]
v_bfe_u32 v16, v20, 7, 1
v_mad_u32_u16 v14, v16, 0x800, v14 op_sel:[0,0,0,0]
v_bfe_u32 v18, v20, 1, 2
v_mad_u32_u16 v13, v18, 32, 0 op_sel:[0,0,0,0]
v_bfe_u32 v19, v20, 3, 1
v_mad_u32_u16 v13, v19, 0x400, v13 op_sel:[0,0,0,0]
v_add_nc_u32 v18, v21, 32
v_bfi_b32 v18, 0xbf, v20, v18
v_bfe_u32 v18, v18, 6, 2
v_mad_u32_u16 v13, v18, 0x80, v13 op_sel:[0,0,0,0]
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
v_readfirstlane_b32 s72, v0
s_and_b32 null, 64, s72
s_cmov_b32 s40, 0x3c00c000
s_lshl_b32 s49, s43, 1
s_lshl_b32 s53, s47, 1
s_lshl_b32 s73, s49, 3
s_lshl_b32 s74, s53, 3
s_and_b32 null, 0x80, s72
s_cselect_b32 s73, s73, 0
s_cselect_b32 s74, s74, 0
s_cselect_b32 s22, 8, 0
s_sub_u32 s22, s9, s22
s_cmov_b32 s22, 0
s_mov_b32 s35, 0x11014000
s_bitcmp1_b32 s14, 4
s_cselect_b32 s75, 0, 0x8000000
s_and_b32 s35, 0xf7ffffff, s35
s_or_b32 s35, s35, s75
s_and_b32 s17, s17, 0xffff
s_add_u32 s17, s17, 0x20000
s_and_b32 s19, s19, 0xffff
s_add_u32 s19, s19, 0x20000
s_add_u32 s16, s16, s73
s_addc_u32 s17, s17, 0
s_add_u32 s18, s18, s74
s_addc_u32 s19, s19, 0
s_mov_b64 s[36:37], s[16:17]
s_mov_b32 s38, 0x80000000
s_mov_b32 s39, 0
s_getpc_b64 s[62:63]
v_cmp_lt_u32 vcc, v0, 0x80
s_cmp_gt_u32 vcc_lo, 0
s_mov_b32 s72, 0x23d8
s_mov_b32 s74, 0x1a58
s_cmov_b32 s72, 0x1e98
s_cmov_b32 s74, 0x1618
s_mov_b32 s73, 0x2654
s_mov_b32 s75, 0x1c54
s_cmov_b32 s73, 0x2114
s_cmov_b32 s75, 0x1814
s_add_u32 s64, s62, s72
s_addc_u32 s65, s63, 0
s_add_u32 s68, s62, s74
s_addc_u32 s69, s63, 0
s_add_u32 s66, s62, s73
s_addc_u32 s67, s63, 0
s_add_u32 s70, s62, s75
s_addc_u32 s71, s63, 0
s_mov_b32 s45, 0
v_mov_b32 v4, 0
s_mov_b32 s56, 0x190
s_bitcmp1_b32 s45, 3
s_cselect_b64 s[62:63], s[64:65], s[68:69]
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
s_swappc_b64 s[62:63], s[62:63]
s_branch 5553
_s_mov_b32__sop1_lit 56, 0x4
s_bitcmp1_b32 s45, 3
s_cselect_b64 s[62:63], s[64:65], s[68:69]
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
s_swappc_b64 s[62:63], s[62:63]
s_branch 5454
s_mov_b32 s56, 0x18c
s_bitcmp1_b32 s45, 4
s_cselect_b64 s[62:63], s[66:67], s[70:71]
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
s_swappc_b64 s[62:63], s[62:63]
s_branch 5380
_s_mov_b32__sop1_lit 56, 0x4
s_bitcmp1_b32 s45, 4
s_cselect_b64 s[62:63], s[66:67], s[70:71]
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
s_swappc_b64 s[62:63], s[62:63]
s_branch 5282
s_mov_b32 s56, 0x190
s_bitcmp1_b32 s45, 3
s_cselect_b64 s[62:63], s[64:65], s[68:69]
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
s_swappc_b64 s[62:63], s[62:63]
s_branch 5207
_s_mov_b32__sop1_lit 56, 0x4
s_bitcmp1_b32 s45, 3
s_cselect_b64 s[62:63], s[64:65], s[68:69]
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
s_swappc_b64 s[62:63], s[62:63]
s_branch 5108
s_mov_b32 s56, 0x18c
s_bitcmp1_b32 s45, 4
s_cselect_b64 s[62:63], s[66:67], s[70:71]
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
s_swappc_b64 s[62:63], s[62:63]
s_branch 5034
_s_mov_b32__sop1_lit 56, 0x4
s_bitcmp1_b32 s45, 4
s_cselect_b64 s[62:63], s[66:67], s[70:71]
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
s_swappc_b64 s[62:63], s[62:63]
s_branch 4936
s_mov_b32 s56, 0x190
s_bitcmp1_b32 s45, 3
s_cselect_b64 s[62:63], s[64:65], s[68:69]
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
s_swappc_b64 s[62:63], s[62:63]
s_branch 4861
_s_mov_b32__sop1_lit 56, 0x4
s_bitcmp1_b32 s45, 3
s_cselect_b64 s[62:63], s[64:65], s[68:69]
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
s_swappc_b64 s[62:63], s[62:63]
s_branch 4762
s_mov_b32 s56, 0x18c
s_bitcmp1_b32 s45, 4
s_cselect_b64 s[62:63], s[66:67], s[70:71]
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
s_swappc_b64 s[62:63], s[62:63]
s_branch 4688
_s_mov_b32__sop1_lit 56, 0x4
s_bitcmp1_b32 s45, 4
s_cselect_b64 s[62:63], s[66:67], s[70:71]
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
s_swappc_b64 s[62:63], s[62:63]
s_branch 4590
s_mov_b32 s56, 0x190
s_bitcmp1_b32 s45, 3
s_cselect_b64 s[62:63], s[64:65], s[68:69]
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
s_swappc_b64 s[62:63], s[62:63]
s_branch 4515
_s_mov_b32__sop1_lit 56, 0x4
s_bitcmp1_b32 s45, 3
s_cselect_b64 s[62:63], s[64:65], s[68:69]
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
s_swappc_b64 s[62:63], s[62:63]
s_branch 4416
s_mov_b32 s56, 0xffffebec
s_bitcmp1_b32 s45, 4
s_cselect_b64 s[62:63], s[66:67], s[70:71]
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
s_swappc_b64 s[62:63], s[62:63]
s_branch 4342
s_mov_b32 s56, 0xffffea64
s_bitcmp1_b32 s45, 4
s_cselect_b64 s[62:63], s[66:67], s[70:71]
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
s_swappc_b64 s[62:63], s[62:63]
s_branch 4244
ds_store_b128 v1, v[18:21] offset:4160
ds_store_b128 v1, v[30:33] offset:16
s_setprio 1
s_ashr_i32 s57, s56, 31
s_add_u32 s62, s62, s56
s_addc_u32 s63, s63, s57
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
ds_load_b128 v[36:39], v11 offset:24768
ds_load_b128 v[40:43], v11 offset:26816
ds_load_b128 v[44:47], v11 offset:28928
ds_load_b128 v[48:51], v11 offset:30976
s_mov_b32 exec_hi, -1
ds_store_b128 v12, v[160:163] offset:16512
s_mov_b32 exec_hi, 0
ds_load_b128 v[53:56], v11 offset:24784
ds_load_b128 v[57:60], v11 offset:26832
ds_load_b128 v[61:64], v11 offset:28944
ds_load_b128 v[65:68], v11 offset:30992
s_mov_b32 exec_hi, -1
ds_store_b128 v12, v[164:167] offset:17024
s_swappc_b64 s[62:63], s[62:63]
ds_store_b128 v2, v[18:21] offset:12416
ds_store_b128 v2, v[30:33] offset:8272
s_setprio 1
s_ashr_i32 s57, s56, 31
s_sub_u32 s23, s23, s34
s_cselect_b64 s[56:57], 0, s[56:57]
s_add_u32 s62, s62, s56
s_addc_u32 s63, s63, s57
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
ds_load_b128 v[7:10], v5 offset:33024
ds_load_b32 v4, v6 offset:35072
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
ds_load_b128 v[36:39], v11 offset:16512
ds_load_b128 v[40:43], v11 offset:18560
ds_load_b128 v[44:47], v11 offset:20672
ds_load_b128 v[48:51], v11 offset:22720
s_mov_b32 exec_hi, -1
ds_store_b128 v12, v[160:163] offset:24768
s_mov_b32 exec_hi, 0
ds_load_b128 v[53:56], v11 offset:16528
ds_load_b128 v[57:60], v11 offset:18576
ds_load_b128 v[61:64], v11 offset:20688
ds_load_b128 v[65:68], v11 offset:22736
s_mov_b32 exec_hi, -1
ds_store_b128 v12, v[164:167] offset:25280
s_waitcnt lgkmcnt(10)
s_swappc_b64 s[62:63], s[62:63]
ds_store_b128 v1, v[18:21] offset:4160
ds_store_b128 v1, v[30:33] offset:16
s_setprio 1
s_ashr_i32 s57, s56, 31
s_add_u32 s62, s62, s56
s_addc_u32 s63, s63, s57
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
ds_load_b128 v[36:39], v11 offset:24768
ds_load_b128 v[40:43], v11 offset:26816
ds_load_b128 v[44:47], v11 offset:28928
ds_load_b128 v[48:51], v11 offset:30976
s_mov_b32 exec_hi, -1
ds_store_b128 v12, v[160:163] offset:16512
s_mov_b32 exec_hi, 0
ds_load_b128 v[53:56], v11 offset:24784
ds_load_b128 v[57:60], v11 offset:26832
ds_load_b128 v[61:64], v11 offset:28944
ds_load_b128 v[65:68], v11 offset:30992
s_mov_b32 exec_hi, -1
ds_store_b128 v12, v[164:167] offset:17024
s_swappc_b64 s[62:63], s[62:63]
ds_store_b128 v2, v[18:21] offset:12416
ds_store_b128 v2, v[30:33] offset:8272
s_setprio 1
s_ashr_i32 s57, s56, 31
s_sub_u32 s23, s23, s34
s_cselect_b64 s[56:57], 0, s[56:57]
s_add_u32 s62, s62, s56
s_addc_u32 s63, s63, s57
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
ds_load_b128 v[7:10], v5 offset:33024
ds_load_b32 v4, v6 offset:35072
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
ds_load_b128 v[36:39], v11 offset:16512
ds_load_b128 v[40:43], v11 offset:18560
ds_load_b128 v[44:47], v11 offset:20672
ds_load_b128 v[48:51], v11 offset:22720
s_mov_b32 exec_hi, -1
ds_store_b128 v12, v[160:163] offset:24768
s_mov_b32 exec_hi, 0
ds_load_b128 v[53:56], v11 offset:16528
ds_load_b128 v[57:60], v11 offset:18576
ds_load_b128 v[61:64], v11 offset:20688
ds_load_b128 v[65:68], v11 offset:22736
s_mov_b32 exec_hi, -1
ds_store_b128 v12, v[164:167] offset:25280
s_waitcnt lgkmcnt(10)
s_swappc_b64 s[62:63], s[62:63]
ds_store_b128 v1, v[18:21] offset:4160
ds_store_b128 v1, v[30:33] offset:16
s_setprio 1
s_ashr_i32 s57, s56, 31
s_add_u32 s62, s62, s56
s_addc_u32 s63, s63, s57
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
ds_load_b128 v[36:39], v11 offset:24768
ds_load_b128 v[40:43], v11 offset:26816
ds_load_b128 v[44:47], v11 offset:28928
ds_load_b128 v[48:51], v11 offset:30976
s_mov_b32 exec_hi, -1
ds_store_b128 v12, v[160:163] offset:16512
s_mov_b32 exec_hi, 0
ds_load_b128 v[53:56], v11 offset:24784
ds_load_b128 v[57:60], v11 offset:26832
ds_load_b128 v[61:64], v11 offset:28944
ds_load_b128 v[65:68], v11 offset:30992
s_mov_b32 exec_hi, -1
ds_store_b128 v12, v[164:167] offset:17024
s_swappc_b64 s[62:63], s[62:63]
ds_store_b128 v2, v[18:21] offset:12416
ds_store_b128 v2, v[30:33] offset:8272
s_setprio 1
s_ashr_i32 s57, s56, 31
s_sub_u32 s23, s23, s34
s_cselect_b64 s[56:57], 0, s[56:57]
s_add_u32 s62, s62, s56
s_addc_u32 s63, s63, s57
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
ds_load_b128 v[7:10], v5 offset:33024
ds_load_b32 v4, v6 offset:35072
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
ds_load_b128 v[36:39], v11 offset:16512
ds_load_b128 v[40:43], v11 offset:18560
ds_load_b128 v[44:47], v11 offset:20672
ds_load_b128 v[48:51], v11 offset:22720
s_mov_b32 exec_hi, -1
ds_store_b128 v12, v[160:163] offset:24768
s_mov_b32 exec_hi, 0
ds_load_b128 v[53:56], v11 offset:16528
ds_load_b128 v[57:60], v11 offset:18576
ds_load_b128 v[61:64], v11 offset:20688
ds_load_b128 v[65:68], v11 offset:22736
s_mov_b32 exec_hi, -1
ds_store_b128 v12, v[164:167] offset:25280
s_waitcnt lgkmcnt(10)
s_swappc_b64 s[62:63], s[62:63]
ds_store_b128 v1, v[18:21] offset:4160
ds_store_b128 v1, v[30:33] offset:16
s_setprio 1
s_ashr_i32 s57, s56, 31
s_add_u32 s62, s62, s56
s_addc_u32 s63, s63, s57
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
ds_load_b128 v[36:39], v11 offset:24768
ds_load_b128 v[40:43], v11 offset:26816
ds_load_b128 v[44:47], v11 offset:28928
ds_load_b128 v[48:51], v11 offset:30976
s_mov_b32 exec_hi, -1
ds_store_b128 v12, v[160:163] offset:16512
s_mov_b32 exec_hi, 0
ds_load_b128 v[53:56], v11 offset:24784
ds_load_b128 v[57:60], v11 offset:26832
ds_load_b128 v[61:64], v11 offset:28944
ds_load_b128 v[65:68], v11 offset:30992
s_mov_b32 exec_hi, -1
ds_store_b128 v12, v[164:167] offset:17024
s_swappc_b64 s[62:63], s[62:63]
ds_store_b128 v2, v[18:21] offset:12416
ds_store_b128 v2, v[30:33] offset:8272
s_setprio 1
s_ashr_i32 s57, s56, 31
s_sub_u32 s23, s23, s34
s_cselect_b64 s[56:57], 0, s[56:57]
s_add_u32 s62, s62, s56
s_addc_u32 s63, s63, s57
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
ds_load_b128 v[7:10], v5 offset:33024
ds_load_b32 v4, v6 offset:35072
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
ds_load_b128 v[36:39], v11 offset:16512
ds_load_b128 v[40:43], v11 offset:18560
ds_load_b128 v[44:47], v11 offset:20672
ds_load_b128 v[48:51], v11 offset:22720
s_mov_b32 exec_hi, -1
ds_store_b128 v12, v[160:163] offset:24768
s_mov_b32 exec_hi, 0
ds_load_b128 v[53:56], v11 offset:16528
ds_load_b128 v[57:60], v11 offset:18576
ds_load_b128 v[61:64], v11 offset:20688
ds_load_b128 v[65:68], v11 offset:22736
s_mov_b32 exec_hi, -1
ds_store_b128 v12, v[164:167] offset:25280
s_waitcnt lgkmcnt(10)
s_swappc_b64 s[62:63], s[62:63]
v_bfe_u32 v32, v0, 6, 1
v_and_b32 v21, 63, v0
v_cmp_eq_u32 vcc, v32, 1
v_cndmask_b32 v42, 0, 0xa00, vcc
v_cndmask_b32 v32, 0, 0x400, vcc
v_cndmask_b32 v37, 0, 0x100, vcc
v_lshl_add_u32 v19, v21, 3, v42
v_lshl_add_u32 v22, v21, 2, v37
v_lshl_add_u32 v23, v21, 2, 0
v_lshl_add_u32 v21, v21, 4, v32
s_cmp_eq_u64 s[30:31], 0
s_cselect_b32 s95, 0, 0x11014000
s_and_b32 s31, s31, 0xffff
s_add_u32 s31, s31, 0x20000
s_mov_b64 s[92:93], s[30:31]
s_mov_b32 s94, 0x80000000
v_and_b32 v32, v0, 63
v_lshlrev_b32 v32, 1, v32
v_cmp_lt_u32 vcc, v32, s12
v_add_nc_u32 v37, v32, 1
v_cndmask_b32 v32, 0x80000000, v32, vcc
v_cmp_lt_u32 vcc, v37, s12
v_cndmask_b32 v37, 0x80000000, v37, vcc
buffer_load_d16_b16 v42, v32, s[92:95], 0 idxen
buffer_load_d16_hi_b16 v42, v37, s[92:95], 0 idxen
s_waitcnt vmcnt(0)
v_readlane_b32 s59, v42, 0
v_readlane_b32 s68, v42, 1
v_readlane_b32 s69, v42, 2
v_readlane_b32 s70, v42, 3
v_readlane_b32 s71, v42, 4
v_readlane_b32 s72, v42, 5
v_readlane_b32 s73, v42, 6
v_readlane_b32 s74, v42, 7
s_mov_b32 s82, 0xbc00c000
s_mov_b32 s40, 0x10000
s_mov_b32 s41, 0x30002
s_mov_b32 s45, 0x10000
v_readfirstlane_b32 s90, v0
s_and_b32 null, 64, s90
s_cmov_b32 s82, 0x3c00c000
s_cmov_b32 s40, 0x20003
s_cmov_b32 s41, 1
s_cmov_b32 s45, 1
s_and_b32 s21, s21, 0xffff
s_add_u32 s21, s21, 0x20000
s_lshl_b32 s80, s51, 1
s_lshl_b32 s81, s52, 1
s_mov_b64 s[76:77], s[20:21]
s_mov_b32 s78, 0x80000000
s_mov_b32 s79, 0
s_bfe_u32 s90, s58, 0x80000
s_cmp_eq_u32 s90, 2
s_cbranch_scc1 23
s_cmp_eq_u32 s90, 0
s_cselect_b32 s32, 1.0, s32
v_cvt_f16_f32 v32, s32
v_readfirstlane_b32 s32, v32
v_cvt_f16_f32 v32, s33
v_readfirstlane_b32 s33, v32
_v_cmp_gt_f16__vop3_s_lit 106, 32, 0x3c00, 0x0, 0x0
s_pack_ll_b32_b16 s32, s32, s32
s_pack_ll_b32_b16 s33, s33, s33
s_cmp_eq_u32 s90, 3
s_cbranch_scc1 7
s_cbranch_vccnz 3
s_mov_b32 s86, 0x4208
s_branch 8
s_mov_b32 s86, 0x45a0
s_branch 5
s_mov_b32 s86, 0x4f50
s_branch 2
s_mov_b32 s86, 0x4938
s_add_u32 s88, s6, 0x3c4c
s_addc_u32 s89, s7, 0
s_sub_u32 s91, s25, 1
s_bitcmp1_b32 s14, 1
s_cselect_b32 s91, s91, 0
s_cselect_b32 s90, -1, 1
s_sub_u32 s93, s24, 1
s_bitcmp1_b32 s14, 0
s_cselect_b32 s93, s93, 0
s_cselect_b32 s92, -1, 1
v_bfe_u32 v47, v0, 6, 1
v_bfe_u32 v52, v0, 4, 1
v_bfe_u32 v32, v0, 5, 1
v_lshl_add_u32 v47, v47, 2, 0
v_lshl_add_u32 v52, v52, 3, v47
v_bfe_u32 v42, v0, 2, 2
v_bfe_u32 v47, v0, 3, 1
v_xor_b32 v37, v0, v0 quad_perm:[0,0,3,1]
v_lshl_add_u32 v32, v32, 1, v52
v_xor_b32 v42, v42, v47
v_add_nc_u32 v47, v32, 1
v_mad_i32_i16 v26, v42, s90, s91 op_sel:[0,0,0,0]
v_mad_i32_i16 v52, v37, s92, s93 op_sel:[0,0,0,0]
v_mad_u32_u16 v26, v52, s48, v26 op_sel:[0,0,0,0]
v_cmp_lt_u32 vcc, v42, s25
v_cndmask_b32 v26, 0x80000000, v26, vcc
v_cmp_lt_u32 vcc, v37, s24
v_cndmask_b32 v26, 0x80000000, v26, vcc
v_mad_u32_u24 v27, v47, s46, v26
v_mad_u32_u24 v26, v32, s46, v26
v_cmp_lt_u32 vcc, v47, s12
v_cndmask_b32 v27, 0x80000000, v27, vcc
v_cmp_lt_u32 vcc, v32, s12
v_cndmask_b32 v26, 0x80000000, v26, vcc
s_add_u32 s91, s28, 1
s_lshr_b32 s91, s91, 1
s_lshl_b32 s92, s91, 1
s_add_u32 s93, s29, 1
s_lshr_b32 s93, s93, 1
s_lshl1_add_u32 s93, s93, 2
s_pack_ll_b32_b16 s22, s93, s91
s_pack_ll_b32_b16 s34, s11, s10
s_sub_u32 s35, s92, s26
s_sub_u32 s90, s93, s27
s_pack_ll_b32_b16 s35, s90, s35
s_pack_ll_b32_b16 s37, s29, s28
s_sub_u32 s90, s93, 1
s_pack_ll_b32_b16 s38, s90, s92
v_lshrrev_b32 v47, 16, s22
v_bfi_b32 v52, 0xffff, s22, 0
v_and_b32 v62, 1, v0
v_bfe_u32 v72, v0, 6, 1
v_and_b32 v37, 63, v0
v_mad_u32_u16 v67, 0x7c, s1, 0 op_sel:[0,0,0,0]
v_mad_u32_u16 v72, 2, s5, v72 op_sel:[0,0,0,0]
v_mad_u32_u16 v57, v47, v52, 0 op_sel:[0,0,0,0]
v_cmp_eq_u32 vcc, 0, v62
v_cndmask_b32 v73, v57, v52, vcc
v_mad_u32_u16 v42, 62, v72, v37 op_sel:[0,0,0,0]
v_cndmask_b32 v42, v67, v42, vcc
v_clz_i32_u32 v78, v73
v_lshlrev_b32 v79, v78, v73
v_and_b32 v80, 0xffffff00, v79
v_cmp_eq_u32 vcc, 0x80000000, v79
v_cvt_f32_u32 v80, v80
v_rcp_f32 v74, v80
v_sub_co_ci_u32 v75, vcc, 32, v78, vcc
v_cvt_f32_ubyte0 v78, v79
v_fma_f32 v80, v80, v74, -1.0
v_fma_f32 v80, v78, v74, v80
v_fmaak_f32 v80, v80, v74, 0x9f000000
v_mul_f32 v80, 0x5f800000, v80
v_mov_b32 v78, 0
v_cvt_floor_i32_f32 v80, -v80
v_lshl_add_u32 v74, v74, 9, v80
v_mad_u64_u32 v[78:79], vcc, v79, v74, v[78:79]
v_sub_co_ci_u32 v74, vcc, v74, -1, vcc
v_mov_b32 v77, v75 quad_perm:[1,1,1,1]
v_mov_b32 v75, v75 quad_perm:[0,0,0,0]
v_mov_b32 v76, v74 quad_perm:[1,1,1,1]
v_mov_b32 v74, v74 quad_perm:[0,0,0,0]
v_mul_hi_u32 v78, v42, v74
v_add_co_u32 v32, vcc, v78, v42
v_add_co_ci_u32 v78, vcc, 0, 0, vcc
v_cmp_eq_u32 vcc, 32, v75
v_cndmask_b32 v32, v32, v78, vcc
v_alignbit_b32 v32, v78, v32, v75
v_mul_hi_u32 v78, v42, v76
v_add_co_u32 v4, vcc, v78, v42
v_add_co_ci_u32 v78, vcc, 0, 0, vcc
v_cmp_eq_u32 vcc, 32, v77
v_cndmask_b32 v4, v4, v78, vcc
v_alignbit_b32 v4, v78, v4, v77
v_mad_u32_u16 v71, v32, v52, 0 op_sel:[0,0,0,0]
v_mad_u32_u16 v70, v4, v47, 0 op_sel:[0,0,0,0]
v_sub_nc_u32 v71, v42, v71
v_sub_nc_u32 v70, v32, v70
v_readlane_b32 s94, v71, 1
v_sub_nc_u32 v71, v71, v52
v_readlane_b32 s23, v70, 1
v_sub_nc_u32 v70, v70, v47
v_readlane_b32 s15, v4, 1
v_sub_nc_u32 v4, v4, s8
s_lshl_b32 s23, s23, 16
s_and_b32 s94, s94, 0xffff
s_add_u32 s23, s23, s94
v_mov_b32 v71, v71 quad_perm:[0,0,2,2]
v_mov_b32 v70, v70 quad_perm:[0,0,2,2]
v_mov_b32 v4, v4 quad_perm:[0,0,2,2]
v_add_co_u32 v71, vcc, v71, v62
v_cndmask_b32 v69, 0, v52, vcc
v_add_co_ci_u32 v70, vcc, v70, 0, vcc
v_cndmask_b32 v68, 0, v47, vcc
v_add_co_ci_u32 v4, vcc, v4, 0, vcc
v_min_u32 v62, v37, 63
v_sub_nc_u32 v71, v71, v69
v_sub_nc_u32 v70, v70, v68
v_cmp_eq_u32 vcc, v37, v62
v_lshlrev_b32 v5, 16, v70
v_bfi_b32 v5, 0xffff, v71, v5
v_add_nc_u32 v81, v4, s8
v_med3_u32 v62, v37, 1, 62
v_mul_lo_u32 v6, v81, s42
v_mul_lo_u32 v11, v81, s50
s_mul_i32 s36, s15, s42
s_mul_i32 s39, s15, s50
v_cndmask_b32 v6, 0x80000000, v6, vcc
v_cmp_eq_u32 vcc, v37, v62
v_cndmask_b32 v11, 0x80000000, v11, vcc
v_cmp_ge_u32 s[54:55], v81, s8
v_cndmask_b32 v6, v6, 0x80000000, s[54:55]
v_cndmask_b32 v11, v11, 0x80000000, s[54:55]
s_mov_b32 s56, 4
s_lshl_b32 s57, s56, 9
v_add_nc_u32 v20, s57, v19
s_bfe_u32 s90, s58, 0x80008
s_bfe_u32 s91, s58, 0x80010
s_cmp_eq_u32 s91, 0
s_cmov_b32 s96, 0
s_cbranch_scc1 108
s_add_u32 s91, s91, 0xffffff00
s_add_u32 s60, s60, 0
s_addc_u32 s61, s61, 0
s_lshr_b32 s95, s13, 2
s_or_b32 s95, s95, 0x21010000
v_cmp_eq_u32 vcc, v0, 0x100
s_cmp_eq_u64 vcc, 0
s_cselect_b32 s95, 0, s95
s_cselect_b32 s94, 0, 0x1010101
s_sub_u32 s90, 0, s90
s_mov_b64 s[92:93], s[60:61]
s_and_b32 s93, s93, 0xffff
s_or_b32 s93, s93, 0x40000
s_and_b32 s99, s22, 0xffff
s_lshr_b32 s98, s22, 16
s_lshr_b32 s99, s99, 1
s_mul_i32 s97, s99, s98
s_mul_i32 s97, s97, s8
s_add_u32 s97, s97, 61
v_writelane_b32 v37, 62, 0
v_writelane_b32 v37, s1, 1
v_writelane_b32 v37, 5, 2
v_clz_i32_u32 v148, v37
v_lshlrev_b32 v149, v148, v37
v_and_b32 v57, 0xffffff00, v149
v_cmp_eq_u32 vcc, 0x80000000, v149
v_cvt_f32_u32 v57, v57
v_rcp_f32 v47, v57
v_sub_co_ci_u32 v52, vcc, 32, v148, vcc
v_cvt_f32_ubyte0 v148, v149
v_fma_f32 v57, v57, v47, -1.0
v_fma_f32 v57, v148, v47, v57
v_fmaak_f32 v57, v57, v47, 0x9f000000
v_mul_f32 v57, 0x5f800000, v57
v_mov_b32 v148, 0
v_cvt_floor_i32_f32 v57, -v57
v_lshl_add_u32 v47, v47, 9, v57
v_mad_u64_u32 v[148:149], vcc, v149, v47, v[148:149]
v_sub_co_ci_u32 v47, vcc, v47, -1, vcc
v_mul_hi_u32 v57, s97, v47
v_add_co_u32 v42, vcc, v57, s97
v_add_co_ci_u32 v57, vcc, 0, 0, vcc
v_cmp_eq_u32 vcc, 32, v52
v_cndmask_b32 v42, v42, v57, vcc
v_alignbit_b32 v42, v57, v42, v52
v_mov_b32 v42, v42 quad_perm:[0,0,0,0]
v_mul_hi_u32 v57, v42, v47
v_add_co_u32 v37, vcc, v57, v42
v_add_co_ci_u32 v57, vcc, 0, 0, vcc
v_cmp_eq_u32 vcc, 32, v52
v_cndmask_b32 v37, v37, v57, vcc
v_alignbit_b32 v37, v57, v37, v52
v_mov_b32 v37, v37 quad_perm:[1,1,1,1]
v_add_nc_u32 v42, v37, 4
v_mul_hi_u32 v57, v42, v47
v_add_co_u32 v42, vcc, v57, v42
v_add_co_ci_u32 v57, vcc, 0, 0, vcc
v_cmp_eq_u32 vcc, 32, v52
v_cndmask_b32 v42, v42, v57, vcc
v_alignbit_b32 v42, v57, v42, v52
v_readlane_b32 s98, v37, 1
v_readlane_b32 s99, v42, 2
s_add_u32 s97, s9, 15
s_lshr_b32 s97, s97, 4
s_cmp_eq_u32 s97, 1
s_cmov_b32 s99, 1
s_add_u32 s96, s98, s99
s_mul_i32 s96, s97, s96
s_add_u32 s96, 5, s96
s_sub_u32 s96, s96, 1
s_mov_b32 s62, 0
s_mov_b32 s63, 0
s_mov_b32 s64, 0
s_mov_b32 s65, 0
s_mov_b32 s66, 0
s_mov_b32 s67, 0
s_mov_b32 s83, 0
s_mov_b32 s49, 8
s_cmp_gt_u32 s83, 0
s_cbranch_scc1 4
v_mov_b32 v15, v4
v_mov_b32 v16, v5
v_mov_b32 v17, v6
v_mov_b32 v18, v11
v_mov_b32 v4, v15
v_mov_b32 v5, v16
v_mov_b32 v6, v17
v_mov_b32 v11, v18
s_add_u32 s83, s83, 16
s_cmp_ge_u32 s83, s9
s_cmov_b32 s83, 0
s_cselect_b32 s84, 6, 2
s_cselect_b32 s85, 9, 0
s_pack_lh_b32_b16 s84, s84, s49
s_pack_ll_b32_b16 s85, s85, s83
v_mov_b32 v14, s84
s_swappc_b64 s[88:89], s[88:89]
s_waitcnt lgkmcnt(0)
s_barrier
v_pk_fma_f16 v48, v53, s82, v48 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v49, v54, s82, v49 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v50, v55, s82, v50 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v51, v56, s82, v51 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_mov_b32 v7, v26
v_mov_b32 v8, v27
v_mov_b32 v9, 0x80000000
v_mov_b32 v10, 0x80000000
v_mov_b32 v12, 0x80000000
v_mov_b32 v13, 0x80000000
s_setprio 0
ds_load_b128 v[38:41], v3
ds_store_b128 v21, v[7:10] offset:33024
ds_load_b128 v[43:46], v3 offset:512
ds_store_b32 v22, v14 offset:35072
s_setprio 2
s_sub_u32 s96, s96, 1
s_cselect_b32 s95, 0x21010000, s95
s_bitcmp1_b32 s62, 2
s_cselect_b32 s88, s86, 0x3c4c
s_add_u32 s88, s6, s88
s_addc_u32 s89, s7, 0
s_swappc_b64 s[88:89], s[88:89]
s_waitcnt lgkmcnt(0)
v_add_nc_u32 v20, s57, v19
v_pk_fma_f16 v151, v38, s82, v28 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v152, v39, s82, v29 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v153, v40, s82, v30 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v154, v41, s82, v31 op_sel:[0,1,0] op_sel_hi:[1,1,1]
_v_pk_add_f16__vop3p 38, 289, 294, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 39, 290, 295, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 40, 291, 296, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 41, 292, 297, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 155, 294, 299, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 156, 295, 300, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 157, 296, 301, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 158, 297, 302, 0x0, 0x3, 0x0, 0x0
s_setprio 0
ds_load_b64 v[24:25], v20 offset:35584
ds_load_b128 v[58:61], v3 offset:2048
ds_load_b128 v[63:66], v3 offset:2560
s_setprio 2
s_mov_b32 s62, s63
s_mov_b32 s63, s64
s_mov_b32 s64, s65
s_mov_b32 s65, s66
s_mov_b32 s66, s67
s_mov_b32 s67, s49
s_bitcmp1_b32 s62, 0
s_cbranch_scc1 2467
s_sub_u32 s56, s56, 1
s_cselect_b32 s56, 4, s56
s_lshl_b32 s57, s56, 9
s_bitcmp1_b32 s62, 1
s_cselect_b32 s88, s87, 0x3c50
s_add_u32 s88, s6, s88
s_addc_u32 s89, s7, 0
s_bitcmp1_b32 s62, 2
s_cselect_b32 s79, 0x11014000, 0
s_sub_u32 s75, s12, 1
s_cselect_b32 s79, 0, s79
s_mov_b64 s[76:77], s[20:21]
s_swappc_b64 s[88:89], s[88:89]
s_waitcnt lgkmcnt(0)
s_barrier
v_pk_fma_f16 v159, v58, s82, v48 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v160, v59, s82, v49 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v161, v60, s82, v50 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v162, v61, s82, v51 op_sel:[0,1,0] op_sel_hi:[1,1,1]
_v_pk_add_f16__vop3p 58, 309, 314, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 59, 310, 315, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 60, 311, 316, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 61, 312, 317, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 163, 314, 319, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 164, 315, 320, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 165, 316, 321, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 166, 317, 322, 0x0, 0x3, 0x0, 0x0
s_add_u32 s91, s91, 0x100
s_cbranch_scc0 7
s_bitset0_b32 s95, 23
s_lshl_b64 exec, 1, s94
buffer_store_b8 v0, off, s[92:95], s4
s_mov_b64 exec, -1
s_mul_i32 s91, s91, 0xffffff01
s_and_not1_b32 null, 0xffffff00, s91
s_cbranch_scc1 3
s_bitset1_b32 s95, 23
buffer_load_b32 v32, off, s[92:95], null glc
s_setprio 0
s_nop 1
ds_load_b128 v[28:31], v3 offset:8256
ds_store_b64 v20, v[12:13] offset:35584
ds_load_b128 v[33:36], v3 offset:8768
ds_load_b32 v14, v23 offset:35072
s_setprio 2
s_bitcmp1_b32 s62, 2
s_cselect_b32 s88, s86, 0x3c4c
s_add_u32 s88, s6, s88
s_addc_u32 s89, s7, 0
s_swappc_b64 s[88:89], s[88:89]
s_waitcnt lgkmcnt(0)
v_readfirstlane_b32 s49, v14
v_pk_fma_f16 v28, v33, s82, v28 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v29, v34, s82, v29 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v30, v35, s82, v30 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v31, v36, s82, v31 op_sel:[0,1,0] op_sel_hi:[1,1,1]
s_setprio 0
ds_load_b128 v[48:51], v3 offset:10304
ds_load_b128 v[53:56], v3 offset:10816
s_setprio 2
s_and_not1_b32 null, 0xffffff00, s91
s_cbranch_scc1 23
s_pack_ll_b32_b16 s90, s90, s90
s_bfm_b64 exec, s95, 0
v_cmp_ne_u32 vcc, v32, s94
s_cbranch_vccz 10
s_add_u32 s90, s90, 0x10000
s_cbranch_scc1 4
buffer_load_b32 v32, off, s[92:95], null glc
s_waitcnt vmcnt(0)
s_branch 65526
s_and_b32 s95, 0xffff0000, s95
s_mov_b32 s90, 0
s_mov_b64 exec, -1
s_mul_i32 s94, s94, 3
s_and_b32 s94, s94, 0x3f3f3f3f
s_add_u32 s92, s92, 0x100
s_and_b32 s92, s92, 0xfffff7ff
s_bitcmp1_b32 s62, 1
s_cselect_b32 s88, s87, 0x3d10
s_add_u32 s88, s6, s88
s_addc_u32 s89, s7, 0
s_cmp_le_u32 s9, 16
s_cselect_b32 s53, -1, 4
s_sub_u32 s53, s53, 1
s_cselect_b32 s84, s85, s84
s_bitset0_b32 s84, 0
s_swappc_b64 s[88:89], s[88:89]
s_waitcnt lgkmcnt(0)
s_barrier
v_pk_fma_f16 v48, v53, s82, v48 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v49, v54, s82, v49 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v50, v55, s82, v50 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v51, v56, s82, v51 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_mov_b32 v14, s84
v_add_co_u32 v149, vcc, v5, s23
v_pk_mad_u16 v167, v5, 0x20001, s35
v_pk_mad_u16 v148, v5, 0x20001, s38
_v_pk_min_u16__vop3p 150, 405, 261, 0x0, 0x3, 0x0, 0x0
v_cndmask_b32 v47, 0, s42, vcc
v_cndmask_b32 v37, 0, s50, vcc
v_mad_u32_u16 v7, v167, 1, v6 op_sel:[0,0,0,0]
v_mad_u32_u16 v12, v148, 1, v11 op_sel:[0,0,0,0]
v_add3_u32 v6, v6, s36, v47
v_add3_u32 v11, v11, s39, v37
_v_pk_sub_u16__vop3p 42, 261, 406, 0x0, 0x3, 0x0, 0x0
v_add_co_ci_u32 v4, s[54:55], v4, s15, vcc
v_cndmask_b32 v6, v6, 0x80000000, s[54:55]
v_cndmask_b32 v11, v11, 0x80000000, s[54:55]
v_cmp_lt_u16 vcc, v167, s34
v_cndmask_b32 v7, 0x80000000, v7, vcc
v_cmp_lt_u16 vcc, v148, s37
v_cndmask_b32 v12, 0x80000000, v12, vcc
_v_pk_ashrrev_i16__vop3p 42, 143, 298, 0x0, 0x2, 0x0, 0x0
_v_pk_add_u16__vop3p 57, 423, 41, 0x1, 0x3, 0x0, 0x0
_v_pk_add_u16__vop3p 52, 423, 40, 0x1, 0x3, 0x0, 0x0
v_mad_u32_u16 v10, v57, s44, v7 op_sel:[1,0,0,0]
v_mad_u32_u16 v8, v52, s44, v7 op_sel:[1,0,0,0]
_v_pk_add_u16__vop3p 62, 404, 45, 0x1, 0x3, 0x0, 0x0
_v_cmp_lt_u16__vop3 106, 57, 34, 0x3
v_cndmask_b32 v10, 0x80000000, v10, vcc
_v_cmp_lt_u16__vop3 106, 52, 34, 0x3
v_cndmask_b32 v8, 0x80000000, v8, vcc
v_mad_u32_u16 v13, v62, s52, v12 op_sel:[1,0,0,0]
v_mad_u32_u16 v9, v57, s44, v7 op_sel:[0,0,0,0]
v_mad_u32_u16 v7, v52, s44, v7 op_sel:[0,0,0,0]
_v_cmp_lt_u16__vop3 106, 62, 37, 0x3
v_cndmask_b32 v13, 0x80000000, v13, vcc
_v_cmp_lt_u16__vop3 106, 57, 34, 0x2
v_cndmask_b32 v9, 0x80000000, v9, vcc
_v_cmp_lt_u16__vop3 106, 52, 34, 0x2
v_cndmask_b32 v7, 0x80000000, v7, vcc
v_mad_u32_u16 v12, v62, s52, v12 op_sel:[0,0,0,0]
v_pk_mad_u16 v5, v42, s22, v149
_v_cmp_lt_u16__vop3 106, 62, 37, 0x2
v_cndmask_b32 v12, 0x80000000, v12, vcc
v_add_co_u32 v37, vcc, v4, s8
v_cndmask_b32 v14, s85, v14, vcc
s_setprio 0
ds_load_b128 v[38:41], v3
ds_store_b128 v21, v[7:10] offset:33024
ds_load_b128 v[43:46], v3 offset:512
ds_store_b32 v22, v14 offset:35072
s_setprio 2
s_sub_u32 s96, s96, 1
s_cselect_b32 s95, 0x21010000, s95
s_bitcmp1_b32 s62, 2
s_cselect_b32 s88, s86, 0x3c4c
s_add_u32 s88, s6, s88
s_addc_u32 s89, s7, 0
s_swappc_b64 s[88:89], s[88:89]
s_waitcnt lgkmcnt(0)
v_add_nc_u32 v20, s57, v19
v_pk_fma_f16 v151, v38, s82, v28 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v152, v39, s82, v29 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v153, v40, s82, v30 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v154, v41, s82, v31 op_sel:[0,1,0] op_sel_hi:[1,1,1]
_v_pk_add_f16__vop3p 38, 289, 294, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 39, 290, 295, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 40, 291, 296, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 41, 292, 297, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 155, 294, 299, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 156, 295, 300, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 157, 296, 301, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 158, 297, 302, 0x0, 0x3, 0x0, 0x0
s_setprio 0
ds_load_b64 v[24:25], v20 offset:35584
ds_load_b128 v[58:61], v3 offset:2048
ds_load_b128 v[63:66], v3 offset:2560
s_setprio 2
s_mov_b32 s62, s63
s_mov_b32 s63, s64
s_mov_b32 s64, s65
s_mov_b32 s65, s66
s_mov_b32 s66, s67
s_mov_b32 s67, s49
s_bitcmp1_b32 s62, 0
s_cbranch_scc1 2179
s_sub_u32 s56, s56, 1
s_cselect_b32 s56, 4, s56
s_lshl_b32 s57, s56, 9
s_bitcmp1_b32 s62, 1
s_cselect_b32 s88, s87, 0x3c50
s_add_u32 s88, s6, s88
s_addc_u32 s89, s7, 0
s_bitcmp1_b32 s62, 2
s_cselect_b32 s79, 0x11014000, 0
s_sub_u32 s75, s12, 1
s_cselect_b32 s79, 0, s79
s_mov_b64 s[76:77], s[20:21]
s_swappc_b64 s[88:89], s[88:89]
s_waitcnt lgkmcnt(0)
s_barrier
v_pk_fma_f16 v159, v58, s82, v48 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v160, v59, s82, v49 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v161, v60, s82, v50 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v162, v61, s82, v51 op_sel:[0,1,0] op_sel_hi:[1,1,1]
_v_pk_add_f16__vop3p 58, 309, 314, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 59, 310, 315, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 60, 311, 316, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 61, 312, 317, 0x0, 0x3, 0x2, 0x2
_v_pk_add_f16__vop3p 163, 314, 319, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 164, 315, 320, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 165, 316, 321, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 166, 317, 322, 0x0, 0x3, 0x0, 0x0
s_add_u32 s91, s91, 0x100
s_cbranch_scc0 7
s_bitset0_b32 s95, 23
s_lshl_b64 exec, 1, s94
buffer_store_b8 v0, off, s[92:95], s4
s_mov_b64 exec, -1
s_mul_i32 s91, s91, 0xffffff01
s_and_not1_b32 null, 0xffffff00, s91
s_cbranch_scc1 3
s_bitset1_b32 s95, 23
buffer_load_b32 v32, off, s[92:95], null glc
s_setprio 0
s_nop 1
ds_load_b128 v[28:31], v3 offset:8256
ds_store_b64 v20, v[12:13] offset:35584
ds_load_b128 v[33:36], v3 offset:8768
ds_load_b32 v14, v23 offset:35072
s_setprio 2
s_bitcmp1_b32 s62, 2
s_cselect_b32 s88, s86, 0x3c4c
s_add_u32 s88, s6, s88
s_addc_u32 s89, s7, 0
s_swappc_b64 s[88:89], s[88:89]
s_waitcnt lgkmcnt(0)
v_readfirstlane_b32 s49, v14
v_pk_fma_f16 v28, v33, s82, v28 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v29, v34, s82, v29 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v30, v35, s82, v30 op_sel:[0,1,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v31, v36, s82, v31 op_sel:[0,1,0] op_sel_hi:[1,1,1]
s_setprio 0
ds_load_b128 v[48:51], v3 offset:10304
ds_load_b128 v[53:56], v3 offset:10816
s_setprio 2
s_and_not1_b32 null, 0xffffff00, s91
s_cbranch_scc1 23
s_pack_ll_b32_b16 s90, s90, s90
s_bfm_b64 exec, s95, 0
v_cmp_ne_u32 vcc, v32, s94
s_cbranch_vccz 10
s_add_u32 s90, s90, 0x10000
s_cbranch_scc1 4
buffer_load_b32 v32, off, s[92:95], null glc
s_waitcnt vmcnt(0)
s_branch 65526
s_and_b32 s95, 0xffff0000, s95
s_mov_b32 s90, 0
s_mov_b64 exec, -1
s_mul_i32 s94, s94, 3
s_and_b32 s94, s94, 0x3f3f3f3f
s_add_u32 s92, s92, 0x100
s_and_b32 s92, s92, 0xfffff7ff
s_bitcmp1_b32 s62, 1
s_cselect_b32 s88, s87, 0x3d10
s_add_u32 s88, s6, s88
s_addc_u32 s89, s7, 0
s_bitcmp1_b32 s49, 1
s_cbranch_scc1 65248
s_branch 65024
s_setpc_b64 s[88:89]
s_bitcmp1_b32 s62, 3
s_cbranch_scc0 40
v_mov_b32 v68, 0
v_mov_b32 v72, 0
v_mov_b32 v69, 0
v_mov_b32 v73, 0
v_mov_b32 v70, 0
v_mov_b32 v74, 0
v_mov_b32 v71, 0
v_mov_b32 v75, 0
v_mov_b32 v84, 0
v_mov_b32 v88, 0
v_mov_b32 v85, 0
v_mov_b32 v89, 0
v_mov_b32 v86, 0
v_mov_b32 v90, 0
v_mov_b32 v87, 0
v_mov_b32 v91, 0
v_mov_b32 v100, 0
v_mov_b32 v104, 0
v_mov_b32 v101, 0
v_mov_b32 v105, 0
v_mov_b32 v102, 0
v_mov_b32 v106, 0
v_mov_b32 v103, 0
v_mov_b32 v107, 0
v_mov_b32 v116, 0
v_mov_b32 v120, 0
v_mov_b32 v117, 0
v_mov_b32 v121, 0
v_mov_b32 v118, 0
v_mov_b32 v122, 0
v_mov_b32 v119, 0
v_mov_b32 v123, 0
v_mov_b32 v132, 0
v_mov_b32 v136, 0
v_mov_b32 v133, 0
v_mov_b32 v137, 0
v_mov_b32 v134, 0
v_mov_b32 v138, 0
v_mov_b32 v135, 0
v_mov_b32 v139, 0
s_mov_b32 s87, 0x3dd0
s_cmp_le_u32 s9, 16
s_cmov_b32 s87, 0x3c4c
s_setpc_b64 s[88:89]
s_bitcmp1_b32 s62, 3
s_cbranch_scc0 40
v_mov_b32 v76, 0
v_mov_b32 v80, 0
v_mov_b32 v77, 0
v_mov_b32 v81, 0
v_mov_b32 v78, 0
v_mov_b32 v82, 0
v_mov_b32 v79, 0
v_mov_b32 v83, 0
v_mov_b32 v92, 0
v_mov_b32 v96, 0
v_mov_b32 v93, 0
v_mov_b32 v97, 0
v_mov_b32 v94, 0
v_mov_b32 v98, 0
v_mov_b32 v95, 0
v_mov_b32 v99, 0
v_mov_b32 v108, 0
v_mov_b32 v112, 0
v_mov_b32 v109, 0
v_mov_b32 v113, 0
v_mov_b32 v110, 0
v_mov_b32 v114, 0
v_mov_b32 v111, 0
v_mov_b32 v115, 0
v_mov_b32 v124, 0
v_mov_b32 v128, 0
v_mov_b32 v125, 0
v_mov_b32 v129, 0
v_mov_b32 v126, 0
v_mov_b32 v130, 0
v_mov_b32 v127, 0
v_mov_b32 v131, 0
v_mov_b32 v140, 0
v_mov_b32 v144, 0
v_mov_b32 v141, 0
v_mov_b32 v145, 0
v_mov_b32 v142, 0
v_mov_b32 v146, 0
v_mov_b32 v143, 0
v_mov_b32 v147, 0
s_mov_b32 s87, 0x3dd0
s_cmp_le_u32 s9, 16
s_cmov_b32 s87, 0x3c4c
s_setpc_b64 s[88:89]
_v_pk_add_f16__vop3p 151, 407, 324, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 152, 408, 325, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 153, 409, 326, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 154, 410, 327, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 155, 411, 328, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 156, 412, 329, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 157, 413, 330, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 158, 414, 331, 0x0, 0x3, 0x0, 0x0
v_mov_b32 v68, v151
v_mov_b32 v69, v152
v_mov_b32 v70, v153
v_mov_b32 v71, v154
v_mov_b32 v72, v155
v_mov_b32 v73, v156
v_mov_b32 v74, v157
v_mov_b32 v75, v158
s_mov_b32 s87, 0x3e3c
s_setpc_b64 s[88:89]
_v_pk_add_f16__vop3p 159, 415, 332, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 160, 416, 333, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 161, 417, 334, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 162, 418, 335, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 163, 419, 336, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 164, 420, 337, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 165, 421, 338, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 166, 422, 339, 0x0, 0x3, 0x0, 0x0
v_mov_b32 v76, v159
v_mov_b32 v77, v160
v_mov_b32 v78, v161
v_mov_b32 v79, v162
v_mov_b32 v80, v163
v_mov_b32 v81, v164
v_mov_b32 v82, v165
v_mov_b32 v83, v166
s_mov_b32 s87, 0x3ea8
s_setpc_b64 s[88:89]
_v_pk_add_f16__vop3p 151, 407, 340, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 152, 408, 341, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 153, 409, 342, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 154, 410, 343, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 155, 411, 344, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 156, 412, 345, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 157, 413, 346, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 158, 414, 347, 0x0, 0x3, 0x0, 0x0
v_mov_b32 v84, v151
v_mov_b32 v85, v152
v_mov_b32 v86, v153
v_mov_b32 v87, v154
v_mov_b32 v88, v155
v_mov_b32 v89, v156
v_mov_b32 v90, v157
v_mov_b32 v91, v158
s_mov_b32 s87, 0x3f14
s_setpc_b64 s[88:89]
_v_pk_add_f16__vop3p 159, 415, 348, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 160, 416, 349, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 161, 417, 350, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 162, 418, 351, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 163, 419, 352, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 164, 420, 353, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 165, 421, 354, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 166, 422, 355, 0x0, 0x3, 0x0, 0x0
v_mov_b32 v92, v159
v_mov_b32 v93, v160
v_mov_b32 v94, v161
v_mov_b32 v95, v162
v_mov_b32 v96, v163
v_mov_b32 v97, v164
v_mov_b32 v98, v165
v_mov_b32 v99, v166
s_mov_b32 s87, 0x3f80
s_setpc_b64 s[88:89]
_v_pk_add_f16__vop3p 151, 407, 356, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 152, 408, 357, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 153, 409, 358, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 154, 410, 359, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 155, 411, 360, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 156, 412, 361, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 157, 413, 362, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 158, 414, 363, 0x0, 0x3, 0x0, 0x0
v_mov_b32 v100, v151
v_mov_b32 v101, v152
v_mov_b32 v102, v153
v_mov_b32 v103, v154
v_mov_b32 v104, v155
v_mov_b32 v105, v156
v_mov_b32 v106, v157
v_mov_b32 v107, v158
s_mov_b32 s87, 0x3fec
s_setpc_b64 s[88:89]
_v_pk_add_f16__vop3p 159, 415, 364, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 160, 416, 365, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 161, 417, 366, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 162, 418, 367, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 163, 419, 368, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 164, 420, 369, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 165, 421, 370, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 166, 422, 371, 0x0, 0x3, 0x0, 0x0
v_mov_b32 v108, v159
v_mov_b32 v109, v160
v_mov_b32 v110, v161
v_mov_b32 v111, v162
v_mov_b32 v112, v163
v_mov_b32 v113, v164
v_mov_b32 v114, v165
v_mov_b32 v115, v166
s_mov_b32 s87, 0x4058
s_setpc_b64 s[88:89]
_v_pk_add_f16__vop3p 151, 407, 372, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 152, 408, 373, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 153, 409, 374, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 154, 410, 375, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 155, 411, 376, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 156, 412, 377, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 157, 413, 378, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 158, 414, 379, 0x0, 0x3, 0x0, 0x0
v_mov_b32 v116, v151
v_mov_b32 v117, v152
v_mov_b32 v118, v153
v_mov_b32 v119, v154
v_mov_b32 v120, v155
v_mov_b32 v121, v156
v_mov_b32 v122, v157
v_mov_b32 v123, v158
s_mov_b32 s87, 0x40c4
s_setpc_b64 s[88:89]
_v_pk_add_f16__vop3p 159, 415, 380, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 160, 416, 381, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 161, 417, 382, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 162, 418, 383, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 163, 419, 384, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 164, 420, 385, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 165, 421, 386, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 166, 422, 387, 0x0, 0x3, 0x0, 0x0
v_mov_b32 v124, v159
v_mov_b32 v125, v160
v_mov_b32 v126, v161
v_mov_b32 v127, v162
v_mov_b32 v128, v163
v_mov_b32 v129, v164
v_mov_b32 v130, v165
v_mov_b32 v131, v166
s_mov_b32 s87, 0x4130
s_setpc_b64 s[88:89]
_v_pk_add_f16__vop3p 151, 407, 388, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 152, 408, 389, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 153, 409, 390, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 154, 410, 391, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 155, 411, 392, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 156, 412, 393, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 157, 413, 394, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 158, 414, 395, 0x0, 0x3, 0x0, 0x0
v_mov_b32 v132, v151
v_mov_b32 v133, v152
v_mov_b32 v134, v153
v_mov_b32 v135, v154
v_mov_b32 v136, v155
v_mov_b32 v137, v156
v_mov_b32 v138, v157
v_mov_b32 v139, v158
s_mov_b32 s87, 0x419c
s_setpc_b64 s[88:89]
_v_pk_add_f16__vop3p 159, 415, 396, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 160, 416, 397, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 161, 417, 398, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 162, 418, 399, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 163, 419, 400, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 164, 420, 401, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 165, 421, 402, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 166, 422, 403, 0x0, 0x3, 0x0, 0x0
v_mov_b32 v140, v159
v_mov_b32 v141, v160
v_mov_b32 v142, v161
v_mov_b32 v143, v162
v_mov_b32 v144, v163
v_mov_b32 v145, v164
v_mov_b32 v146, v165
v_mov_b32 v147, v166
s_mov_b32 s87, 0x3dd0
s_setpc_b64 s[88:89]
_v_pk_add_f16__vop3p 151, 407, 59, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 152, 408, 68, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 153, 409, 69, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 154, 410, 70, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 155, 411, 59, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 156, 412, 68, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 157, 413, 69, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 158, 414, 70, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 37, 32, 407, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 47, 32, 408, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 57, 32, 409, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 67, 32, 410, 0x0, 0x3, 0x0, 0x0
_v_pk_max_f16__vop3p 151, 407, 293, 0x0, 0x3, 0x0, 0x0
_v_pk_max_f16__vop3p 152, 408, 303, 0x0, 0x3, 0x0, 0x0
_v_pk_max_f16__vop3p 153, 409, 313, 0x0, 0x3, 0x0, 0x0
_v_pk_max_f16__vop3p 154, 410, 323, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 37, 32, 411, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 47, 32, 412, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 57, 32, 413, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 67, 32, 414, 0x0, 0x3, 0x0, 0x0
_v_pk_max_f16__vop3p 155, 411, 293, 0x0, 0x3, 0x0, 0x0
_v_pk_max_f16__vop3p 156, 412, 303, 0x0, 0x3, 0x0, 0x0
_v_pk_max_f16__vop3p 157, 413, 313, 0x0, 0x3, 0x0, 0x0
_v_pk_max_f16__vop3p 158, 414, 323, 0x0, 0x3, 0x0, 0x0
s_mov_b32 s86, 0x42d4
s_setpc_b64 s[88:89]
_v_pk_add_f16__vop3p 159, 415, 71, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 160, 416, 72, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 161, 417, 73, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 162, 418, 74, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 163, 419, 71, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 164, 420, 72, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 165, 421, 73, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 166, 422, 74, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 37, 32, 415, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 47, 32, 416, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 57, 32, 417, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 67, 32, 418, 0x0, 0x3, 0x0, 0x0
_v_pk_max_f16__vop3p 159, 415, 293, 0x0, 0x3, 0x0, 0x0
_v_pk_max_f16__vop3p 160, 416, 303, 0x0, 0x3, 0x0, 0x0
_v_pk_max_f16__vop3p 161, 417, 313, 0x0, 0x3, 0x0, 0x0
_v_pk_max_f16__vop3p 162, 418, 323, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 37, 32, 419, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 47, 32, 420, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 57, 32, 421, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 67, 32, 422, 0x0, 0x3, 0x0, 0x0
_v_pk_max_f16__vop3p 163, 419, 293, 0x0, 0x3, 0x0, 0x0
_v_pk_max_f16__vop3p 164, 420, 303, 0x0, 0x3, 0x0, 0x0
_v_pk_max_f16__vop3p 165, 421, 313, 0x0, 0x3, 0x0, 0x0
_v_pk_max_f16__vop3p 166, 422, 323, 0x0, 0x3, 0x0, 0x0
buffer_store_b16 v151, v24, s[76:79], 0 idxen
buffer_store_b16 v155, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_d16_hi_b16 v151, v24, s[76:79], 0 idxen
buffer_store_d16_hi_b16 v155, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_b16 v152, v24, s[76:79], 0 idxen
buffer_store_b16 v156, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_d16_hi_b16 v152, v24, s[76:79], 0 idxen
buffer_store_d16_hi_b16 v156, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_b16 v153, v24, s[76:79], 0 idxen
buffer_store_b16 v157, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_d16_hi_b16 v153, v24, s[76:79], 0 idxen
buffer_store_d16_hi_b16 v157, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_b16 v154, v24, s[76:79], 0 idxen
buffer_store_b16 v158, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_d16_hi_b16 v154, v24, s[76:79], 0 idxen
buffer_store_d16_hi_b16 v158, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_b16 v159, v24, s[76:79], 0 idxen
buffer_store_b16 v163, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_d16_hi_b16 v159, v24, s[76:79], 0 idxen
buffer_store_d16_hi_b16 v163, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_b16 v160, v24, s[76:79], 0 idxen
buffer_store_b16 v164, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_d16_hi_b16 v160, v24, s[76:79], 0 idxen
buffer_store_d16_hi_b16 v164, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_b16 v161, v24, s[76:79], 0 idxen
buffer_store_b16 v165, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_d16_hi_b16 v161, v24, s[76:79], 0 idxen
buffer_store_d16_hi_b16 v165, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_b16 v162, v24, s[76:79], 0 idxen
buffer_store_b16 v166, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_d16_hi_b16 v162, v24, s[76:79], 0 idxen
buffer_store_d16_hi_b16 v166, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
s_mov_b32 s86, 0x4208
s_setpc_b64 s[88:89]
_v_pk_add_f16__vop3p 151, 407, 59, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 152, 408, 68, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 153, 409, 69, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 154, 410, 70, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 155, 411, 59, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 156, 412, 68, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 157, 413, 69, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 158, 414, 70, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 37, 32, 407, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 47, 32, 408, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 57, 32, 409, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 67, 32, 410, 0x0, 0x3, 0x0, 0x0
_v_pk_min_f16__vop3p 151, 407, 293, 0x0, 0x3, 0x0, 0x0
_v_pk_min_f16__vop3p 152, 408, 303, 0x0, 0x3, 0x0, 0x0
_v_pk_min_f16__vop3p 153, 409, 313, 0x0, 0x3, 0x0, 0x0
_v_pk_min_f16__vop3p 154, 410, 323, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 37, 32, 411, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 47, 32, 412, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 57, 32, 413, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 67, 32, 414, 0x0, 0x3, 0x0, 0x0
_v_pk_min_f16__vop3p 155, 411, 293, 0x0, 0x3, 0x0, 0x0
_v_pk_min_f16__vop3p 156, 412, 303, 0x0, 0x3, 0x0, 0x0
_v_pk_min_f16__vop3p 157, 413, 313, 0x0, 0x3, 0x0, 0x0
_v_pk_min_f16__vop3p 158, 414, 323, 0x0, 0x3, 0x0, 0x0
s_mov_b32 s86, 0x466c
s_setpc_b64 s[88:89]
_v_pk_add_f16__vop3p 159, 415, 71, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 160, 416, 72, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 161, 417, 73, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 162, 418, 74, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 163, 419, 71, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 164, 420, 72, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 165, 421, 73, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 166, 422, 74, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 37, 32, 415, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 47, 32, 416, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 57, 32, 417, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 67, 32, 418, 0x0, 0x3, 0x0, 0x0
_v_pk_min_f16__vop3p 159, 415, 293, 0x0, 0x3, 0x0, 0x0
_v_pk_min_f16__vop3p 160, 416, 303, 0x0, 0x3, 0x0, 0x0
_v_pk_min_f16__vop3p 161, 417, 313, 0x0, 0x3, 0x0, 0x0
_v_pk_min_f16__vop3p 162, 418, 323, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 37, 32, 419, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 47, 32, 420, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 57, 32, 421, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 67, 32, 422, 0x0, 0x3, 0x0, 0x0
_v_pk_min_f16__vop3p 163, 419, 293, 0x0, 0x3, 0x0, 0x0
_v_pk_min_f16__vop3p 164, 420, 303, 0x0, 0x3, 0x0, 0x0
_v_pk_min_f16__vop3p 165, 421, 313, 0x0, 0x3, 0x0, 0x0
_v_pk_min_f16__vop3p 166, 422, 323, 0x0, 0x3, 0x0, 0x0
buffer_store_b16 v151, v24, s[76:79], 0 idxen
buffer_store_b16 v155, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_d16_hi_b16 v151, v24, s[76:79], 0 idxen
buffer_store_d16_hi_b16 v155, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_b16 v152, v24, s[76:79], 0 idxen
buffer_store_b16 v156, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_d16_hi_b16 v152, v24, s[76:79], 0 idxen
buffer_store_d16_hi_b16 v156, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_b16 v153, v24, s[76:79], 0 idxen
buffer_store_b16 v157, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_d16_hi_b16 v153, v24, s[76:79], 0 idxen
buffer_store_d16_hi_b16 v157, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_b16 v154, v24, s[76:79], 0 idxen
buffer_store_b16 v158, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_d16_hi_b16 v154, v24, s[76:79], 0 idxen
buffer_store_d16_hi_b16 v158, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_b16 v159, v24, s[76:79], 0 idxen
buffer_store_b16 v163, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_d16_hi_b16 v159, v24, s[76:79], 0 idxen
buffer_store_d16_hi_b16 v163, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_b16 v160, v24, s[76:79], 0 idxen
buffer_store_b16 v164, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_d16_hi_b16 v160, v24, s[76:79], 0 idxen
buffer_store_d16_hi_b16 v164, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_b16 v161, v24, s[76:79], 0 idxen
buffer_store_b16 v165, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_d16_hi_b16 v161, v24, s[76:79], 0 idxen
buffer_store_d16_hi_b16 v165, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_b16 v162, v24, s[76:79], 0 idxen
buffer_store_b16 v166, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_d16_hi_b16 v162, v24, s[76:79], 0 idxen
buffer_store_d16_hi_b16 v166, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
s_mov_b32 s86, 0x45a0
s_setpc_b64 s[88:89]
_v_pk_add_f16__vop3p 151, 407, 59, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 152, 408, 68, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 153, 409, 69, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 154, 410, 70, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 155, 411, 59, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 156, 412, 68, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 157, 413, 69, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 158, 414, 70, 0x0, 0x3, 0x0, 0x0
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
s_mov_b32 s86, 0x4b44
s_setpc_b64 s[88:89]
_v_pk_add_f16__vop3p 159, 415, 71, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 160, 416, 72, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 161, 417, 73, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 162, 418, 74, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 163, 419, 71, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 164, 420, 72, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 165, 421, 73, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 166, 422, 74, 0x0, 0x3, 0x0, 0x0
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
_v_pk_mul_f16__vop3p_lit 163, 0xbdc5bdc5, 419, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 164, 0xbdc5bdc5, 420, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 165, 0xbdc5bdc5, 421, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 166, 0xbdc5bdc5, 422, 0x0, 0x3
v_exp_f16 v163, v163
v_exp_f16 v164, v164
v_exp_f16 v165, v165
v_exp_f16 v166, v166
_v_exp_f16__vop3 163, 163, 0x9
_v_exp_f16__vop3 164, 164, 0x9
_v_exp_f16__vop3 165, 165, 0x9
_v_exp_f16__vop3 166, 166, 0x9
_v_pk_add_f16__vop3p_lit 163, 0x3c003c00, 419, 0x0, 0x3
_v_pk_add_f16__vop3p_lit 164, 0x3c003c00, 420, 0x0, 0x3
_v_pk_add_f16__vop3p_lit 165, 0x3c003c00, 421, 0x0, 0x3
_v_pk_add_f16__vop3p_lit 166, 0x3c003c00, 422, 0x0, 0x3
v_rcp_f16 v163, v163
v_rcp_f16 v164, v164
v_rcp_f16 v165, v165
v_rcp_f16 v166, v166
_v_rcp_f16__vop3 163, 163, 0x9
_v_rcp_f16__vop3 164, 164, 0x9
_v_rcp_f16__vop3 165, 165, 0x9
_v_rcp_f16__vop3 166, 166, 0x9
buffer_store_b16 v151, v24, s[76:79], 0 idxen
buffer_store_b16 v155, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_d16_hi_b16 v151, v24, s[76:79], 0 idxen
buffer_store_d16_hi_b16 v155, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_b16 v152, v24, s[76:79], 0 idxen
buffer_store_b16 v156, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_d16_hi_b16 v152, v24, s[76:79], 0 idxen
buffer_store_d16_hi_b16 v156, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_b16 v153, v24, s[76:79], 0 idxen
buffer_store_b16 v157, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_d16_hi_b16 v153, v24, s[76:79], 0 idxen
buffer_store_d16_hi_b16 v157, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_b16 v154, v24, s[76:79], 0 idxen
buffer_store_b16 v158, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_d16_hi_b16 v154, v24, s[76:79], 0 idxen
buffer_store_d16_hi_b16 v158, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_b16 v159, v24, s[76:79], 0 idxen
buffer_store_b16 v163, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_d16_hi_b16 v159, v24, s[76:79], 0 idxen
buffer_store_d16_hi_b16 v163, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_b16 v160, v24, s[76:79], 0 idxen
buffer_store_b16 v164, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_d16_hi_b16 v160, v24, s[76:79], 0 idxen
buffer_store_d16_hi_b16 v164, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_b16 v161, v24, s[76:79], 0 idxen
buffer_store_b16 v165, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_d16_hi_b16 v161, v24, s[76:79], 0 idxen
buffer_store_d16_hi_b16 v165, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_b16 v162, v24, s[76:79], 0 idxen
buffer_store_b16 v166, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_d16_hi_b16 v162, v24, s[76:79], 0 idxen
buffer_store_d16_hi_b16 v166, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
s_mov_b32 s86, 0x4938
s_setpc_b64 s[88:89]
_v_pk_add_f16__vop3p 151, 407, 59, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 152, 408, 68, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 153, 409, 69, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 154, 410, 70, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 155, 411, 59, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 156, 412, 68, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 157, 413, 69, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 158, 414, 70, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 151, 407, 33, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 152, 408, 33, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 153, 409, 33, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 154, 410, 33, 0x0, 0x3, 0x0, 0x0
v_and_b32 v37, 0x7fff7fff, v151
v_and_b32 v47, 0x7fff7fff, v152
v_and_b32 v57, 0x7fff7fff, v153
v_and_b32 v67, 0x7fff7fff, v154
v_mov_b32 v42, 0xb5f8b5f8
v_mov_b32 v52, 0xb5f8b5f8
v_mov_b32 v62, 0xb5f8b5f8
v_mov_b32 v148, 0xb5f8b5f8
v_pk_fma_f16 v42, v37, 0x2ff12ff1, v42
v_pk_fma_f16 v52, v47, 0x2ff12ff1, v52
v_pk_fma_f16 v62, v57, 0x2ff12ff1, v62
v_pk_fma_f16 v148, v67, 0x2ff12ff1, v148
v_pk_fma_f16 v42, v37, v42, 0x1c571c57
v_pk_fma_f16 v52, v47, v52, 0x1c571c57
v_pk_fma_f16 v62, v57, v62, 0x1c571c57
v_pk_fma_f16 v148, v67, v148, 0x1c571c57
v_pk_fma_f16 v42, v37, v42, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_pk_fma_f16 v52, v47, v52, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_pk_fma_f16 v62, v57, v62, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_pk_fma_f16 v148, v67, v148, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
_v_pk_mul_f16__vop3p 42, 293, 298, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 52, 303, 308, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 62, 313, 318, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 148, 323, 404, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p_lit 37, 0x41c541c5, 293, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 47, 0x41c541c5, 303, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 57, 0x41c541c5, 313, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 67, 0x41c541c5, 323, 0x0, 0x3
v_exp_f16 v37, v37
v_exp_f16 v47, v47
v_exp_f16 v57, v57
v_exp_f16 v67, v67
_v_exp_f16__vop1 (37 | /*op_sel*/ 0x80), (37 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (47 | /*op_sel*/ 0x80), (47 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (57 | /*op_sel*/ 0x80), (57 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (67 | /*op_sel*/ 0x80), (67 | /*op_sel*/ 0x80)
_v_pk_add_f16__vop3p 37, 242, 293, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 47, 242, 303, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 57, 242, 313, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 67, 242, 323, 0x0, 0x2, 0x0, 0x0
v_rcp_f16 v37, v37
v_rcp_f16 v47, v47
v_rcp_f16 v57, v57
v_rcp_f16 v67, v67
_v_rcp_f16__vop1 (37 | /*op_sel*/ 0x80), (37 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (47 | /*op_sel*/ 0x80), (47 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (57 | /*op_sel*/ 0x80), (57 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (67 | /*op_sel*/ 0x80), (67 | /*op_sel*/ 0x80)
v_pk_fma_f16 v37, v37, 2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0] neg_lo:[1,0,0] neg_hi:[1,0,0]
v_pk_fma_f16 v47, v47, 2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0] neg_lo:[1,0,0] neg_hi:[1,0,0]
v_pk_fma_f16 v57, v57, 2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0] neg_lo:[1,0,0] neg_hi:[1,0,0]
v_pk_fma_f16 v67, v67, 2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0] neg_lo:[1,0,0] neg_hi:[1,0,0]
_v_cmp_gt_f16__vop3_v_lit 106, 151, 0x38b838b8, 0x0, 0x1
v_cndmask_b16 v42, v42, v37, vcc
_v_cmp_gt_f16__vop3_v_lit 106, 152, 0x38b838b8, 0x0, 0x1
v_cndmask_b16 v52, v52, v47, vcc
_v_cmp_gt_f16__vop3_v_lit 106, 153, 0x38b838b8, 0x0, 0x1
v_cndmask_b16 v62, v62, v57, vcc
_v_cmp_gt_f16__vop3_v_lit 106, 154, 0x38b838b8, 0x0, 0x1
v_cndmask_b16 v148, v148, v67, vcc
_v_cmp_gt_f16__vop3_v_lit 106, 151, 0x38b838b8, 0x3, 0x1
_v_cndmask_b16__vop3 42, 42, 37, 106, 0xb
_v_cmp_gt_f16__vop3_v_lit 106, 152, 0x38b838b8, 0x3, 0x1
_v_cndmask_b16__vop3 52, 52, 47, 106, 0xb
_v_cmp_gt_f16__vop3_v_lit 106, 153, 0x38b838b8, 0x3, 0x1
_v_cndmask_b16__vop3 62, 62, 57, 106, 0xb
_v_cmp_gt_f16__vop3_v_lit 106, 154, 0x38b838b8, 0x3, 0x1
_v_cndmask_b16__vop3 148, 148, 67, 106, 0xb
v_bfi_b32 v151, 0x7fff7fff, v42, v151
v_bfi_b32 v152, 0x7fff7fff, v52, v152
v_bfi_b32 v153, 0x7fff7fff, v62, v153
v_bfi_b32 v154, 0x7fff7fff, v148, v154
_v_pk_mul_f16__vop3p 151, 407, 32, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 152, 408, 32, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 153, 409, 32, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 154, 410, 32, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 155, 411, 33, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 156, 412, 33, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 157, 413, 33, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 158, 414, 33, 0x0, 0x3, 0x0, 0x0
v_and_b32 v37, 0x7fff7fff, v155
v_and_b32 v47, 0x7fff7fff, v156
v_and_b32 v57, 0x7fff7fff, v157
v_and_b32 v67, 0x7fff7fff, v158
v_mov_b32 v42, 0xb5f8b5f8
v_mov_b32 v52, 0xb5f8b5f8
v_mov_b32 v62, 0xb5f8b5f8
v_mov_b32 v148, 0xb5f8b5f8
v_pk_fma_f16 v42, v37, 0x2ff12ff1, v42
v_pk_fma_f16 v52, v47, 0x2ff12ff1, v52
v_pk_fma_f16 v62, v57, 0x2ff12ff1, v62
v_pk_fma_f16 v148, v67, 0x2ff12ff1, v148
v_pk_fma_f16 v42, v37, v42, 0x1c571c57
v_pk_fma_f16 v52, v47, v52, 0x1c571c57
v_pk_fma_f16 v62, v57, v62, 0x1c571c57
v_pk_fma_f16 v148, v67, v148, 0x1c571c57
v_pk_fma_f16 v42, v37, v42, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_pk_fma_f16 v52, v47, v52, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_pk_fma_f16 v62, v57, v62, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_pk_fma_f16 v148, v67, v148, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
_v_pk_mul_f16__vop3p 42, 293, 298, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 52, 303, 308, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 62, 313, 318, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 148, 323, 404, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p_lit 37, 0x41c541c5, 293, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 47, 0x41c541c5, 303, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 57, 0x41c541c5, 313, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 67, 0x41c541c5, 323, 0x0, 0x3
v_exp_f16 v37, v37
v_exp_f16 v47, v47
v_exp_f16 v57, v57
v_exp_f16 v67, v67
_v_exp_f16__vop1 (37 | /*op_sel*/ 0x80), (37 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (47 | /*op_sel*/ 0x80), (47 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (57 | /*op_sel*/ 0x80), (57 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (67 | /*op_sel*/ 0x80), (67 | /*op_sel*/ 0x80)
_v_pk_add_f16__vop3p 37, 242, 293, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 47, 242, 303, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 57, 242, 313, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 67, 242, 323, 0x0, 0x2, 0x0, 0x0
v_rcp_f16 v37, v37
v_rcp_f16 v47, v47
v_rcp_f16 v57, v57
v_rcp_f16 v67, v67
_v_rcp_f16__vop1 (37 | /*op_sel*/ 0x80), (37 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (47 | /*op_sel*/ 0x80), (47 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (57 | /*op_sel*/ 0x80), (57 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (67 | /*op_sel*/ 0x80), (67 | /*op_sel*/ 0x80)
v_pk_fma_f16 v37, v37, 2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0] neg_lo:[1,0,0] neg_hi:[1,0,0]
v_pk_fma_f16 v47, v47, 2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0] neg_lo:[1,0,0] neg_hi:[1,0,0]
v_pk_fma_f16 v57, v57, 2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0] neg_lo:[1,0,0] neg_hi:[1,0,0]
v_pk_fma_f16 v67, v67, 2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0] neg_lo:[1,0,0] neg_hi:[1,0,0]
_v_cmp_gt_f16__vop3_v_lit 106, 155, 0x38b838b8, 0x0, 0x1
v_cndmask_b16 v42, v42, v37, vcc
_v_cmp_gt_f16__vop3_v_lit 106, 156, 0x38b838b8, 0x0, 0x1
v_cndmask_b16 v52, v52, v47, vcc
_v_cmp_gt_f16__vop3_v_lit 106, 157, 0x38b838b8, 0x0, 0x1
v_cndmask_b16 v62, v62, v57, vcc
_v_cmp_gt_f16__vop3_v_lit 106, 158, 0x38b838b8, 0x0, 0x1
v_cndmask_b16 v148, v148, v67, vcc
_v_cmp_gt_f16__vop3_v_lit 106, 155, 0x38b838b8, 0x3, 0x1
_v_cndmask_b16__vop3 42, 42, 37, 106, 0xb
_v_cmp_gt_f16__vop3_v_lit 106, 156, 0x38b838b8, 0x3, 0x1
_v_cndmask_b16__vop3 52, 52, 47, 106, 0xb
_v_cmp_gt_f16__vop3_v_lit 106, 157, 0x38b838b8, 0x3, 0x1
_v_cndmask_b16__vop3 62, 62, 57, 106, 0xb
_v_cmp_gt_f16__vop3_v_lit 106, 158, 0x38b838b8, 0x3, 0x1
_v_cndmask_b16__vop3 148, 148, 67, 106, 0xb
v_bfi_b32 v155, 0x7fff7fff, v42, v155
v_bfi_b32 v156, 0x7fff7fff, v52, v156
v_bfi_b32 v157, 0x7fff7fff, v62, v157
v_bfi_b32 v158, 0x7fff7fff, v148, v158
_v_pk_mul_f16__vop3p 155, 411, 32, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 156, 412, 32, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 157, 413, 32, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 158, 414, 32, 0x0, 0x3, 0x0, 0x0
s_mov_b32 s86, 0x54dc
s_setpc_b64 s[88:89]
_v_pk_add_f16__vop3p 159, 415, 71, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 160, 416, 72, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 161, 417, 73, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 162, 418, 74, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 163, 419, 71, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 164, 420, 72, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 165, 421, 73, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 166, 422, 74, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 159, 415, 33, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 160, 416, 33, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 161, 417, 33, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 162, 418, 33, 0x0, 0x3, 0x0, 0x0
v_and_b32 v37, 0x7fff7fff, v159
v_and_b32 v47, 0x7fff7fff, v160
v_and_b32 v57, 0x7fff7fff, v161
v_and_b32 v67, 0x7fff7fff, v162
v_mov_b32 v42, 0xb5f8b5f8
v_mov_b32 v52, 0xb5f8b5f8
v_mov_b32 v62, 0xb5f8b5f8
v_mov_b32 v148, 0xb5f8b5f8
v_pk_fma_f16 v42, v37, 0x2ff12ff1, v42
v_pk_fma_f16 v52, v47, 0x2ff12ff1, v52
v_pk_fma_f16 v62, v57, 0x2ff12ff1, v62
v_pk_fma_f16 v148, v67, 0x2ff12ff1, v148
v_pk_fma_f16 v42, v37, v42, 0x1c571c57
v_pk_fma_f16 v52, v47, v52, 0x1c571c57
v_pk_fma_f16 v62, v57, v62, 0x1c571c57
v_pk_fma_f16 v148, v67, v148, 0x1c571c57
v_pk_fma_f16 v42, v37, v42, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_pk_fma_f16 v52, v47, v52, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_pk_fma_f16 v62, v57, v62, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_pk_fma_f16 v148, v67, v148, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
_v_pk_mul_f16__vop3p 42, 293, 298, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 52, 303, 308, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 62, 313, 318, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 148, 323, 404, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p_lit 37, 0x41c541c5, 293, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 47, 0x41c541c5, 303, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 57, 0x41c541c5, 313, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 67, 0x41c541c5, 323, 0x0, 0x3
v_exp_f16 v37, v37
v_exp_f16 v47, v47
v_exp_f16 v57, v57
v_exp_f16 v67, v67
_v_exp_f16__vop1 (37 | /*op_sel*/ 0x80), (37 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (47 | /*op_sel*/ 0x80), (47 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (57 | /*op_sel*/ 0x80), (57 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (67 | /*op_sel*/ 0x80), (67 | /*op_sel*/ 0x80)
_v_pk_add_f16__vop3p 37, 242, 293, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 47, 242, 303, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 57, 242, 313, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 67, 242, 323, 0x0, 0x2, 0x0, 0x0
v_rcp_f16 v37, v37
v_rcp_f16 v47, v47
v_rcp_f16 v57, v57
v_rcp_f16 v67, v67
_v_rcp_f16__vop1 (37 | /*op_sel*/ 0x80), (37 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (47 | /*op_sel*/ 0x80), (47 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (57 | /*op_sel*/ 0x80), (57 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (67 | /*op_sel*/ 0x80), (67 | /*op_sel*/ 0x80)
v_pk_fma_f16 v37, v37, 2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0] neg_lo:[1,0,0] neg_hi:[1,0,0]
v_pk_fma_f16 v47, v47, 2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0] neg_lo:[1,0,0] neg_hi:[1,0,0]
v_pk_fma_f16 v57, v57, 2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0] neg_lo:[1,0,0] neg_hi:[1,0,0]
v_pk_fma_f16 v67, v67, 2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0] neg_lo:[1,0,0] neg_hi:[1,0,0]
_v_cmp_gt_f16__vop3_v_lit 106, 159, 0x38b838b8, 0x0, 0x1
v_cndmask_b16 v42, v42, v37, vcc
_v_cmp_gt_f16__vop3_v_lit 106, 160, 0x38b838b8, 0x0, 0x1
v_cndmask_b16 v52, v52, v47, vcc
_v_cmp_gt_f16__vop3_v_lit 106, 161, 0x38b838b8, 0x0, 0x1
v_cndmask_b16 v62, v62, v57, vcc
_v_cmp_gt_f16__vop3_v_lit 106, 162, 0x38b838b8, 0x0, 0x1
v_cndmask_b16 v148, v148, v67, vcc
_v_cmp_gt_f16__vop3_v_lit 106, 159, 0x38b838b8, 0x3, 0x1
_v_cndmask_b16__vop3 42, 42, 37, 106, 0xb
_v_cmp_gt_f16__vop3_v_lit 106, 160, 0x38b838b8, 0x3, 0x1
_v_cndmask_b16__vop3 52, 52, 47, 106, 0xb
_v_cmp_gt_f16__vop3_v_lit 106, 161, 0x38b838b8, 0x3, 0x1
_v_cndmask_b16__vop3 62, 62, 57, 106, 0xb
_v_cmp_gt_f16__vop3_v_lit 106, 162, 0x38b838b8, 0x3, 0x1
_v_cndmask_b16__vop3 148, 148, 67, 106, 0xb
v_bfi_b32 v159, 0x7fff7fff, v42, v159
v_bfi_b32 v160, 0x7fff7fff, v52, v160
v_bfi_b32 v161, 0x7fff7fff, v62, v161
v_bfi_b32 v162, 0x7fff7fff, v148, v162
_v_pk_mul_f16__vop3p 159, 415, 32, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 160, 416, 32, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 161, 417, 32, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 162, 418, 32, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 163, 419, 33, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 164, 420, 33, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 165, 421, 33, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 166, 422, 33, 0x0, 0x3, 0x0, 0x0
v_and_b32 v37, 0x7fff7fff, v163
v_and_b32 v47, 0x7fff7fff, v164
v_and_b32 v57, 0x7fff7fff, v165
v_and_b32 v67, 0x7fff7fff, v166
v_mov_b32 v42, 0xb5f8b5f8
v_mov_b32 v52, 0xb5f8b5f8
v_mov_b32 v62, 0xb5f8b5f8
v_mov_b32 v148, 0xb5f8b5f8
v_pk_fma_f16 v42, v37, 0x2ff12ff1, v42
v_pk_fma_f16 v52, v47, 0x2ff12ff1, v52
v_pk_fma_f16 v62, v57, 0x2ff12ff1, v62
v_pk_fma_f16 v148, v67, 0x2ff12ff1, v148
v_pk_fma_f16 v42, v37, v42, 0x1c571c57
v_pk_fma_f16 v52, v47, v52, 0x1c571c57
v_pk_fma_f16 v62, v57, v62, 0x1c571c57
v_pk_fma_f16 v148, v67, v148, 0x1c571c57
v_pk_fma_f16 v42, v37, v42, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_pk_fma_f16 v52, v47, v52, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_pk_fma_f16 v62, v57, v62, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
v_pk_fma_f16 v148, v67, v148, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0]
_v_pk_mul_f16__vop3p 42, 293, 298, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 52, 303, 308, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 62, 313, 318, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 148, 323, 404, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p_lit 37, 0x41c541c5, 293, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 47, 0x41c541c5, 303, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 57, 0x41c541c5, 313, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 67, 0x41c541c5, 323, 0x0, 0x3
v_exp_f16 v37, v37
v_exp_f16 v47, v47
v_exp_f16 v57, v57
v_exp_f16 v67, v67
_v_exp_f16__vop1 (37 | /*op_sel*/ 0x80), (37 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (47 | /*op_sel*/ 0x80), (47 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (57 | /*op_sel*/ 0x80), (57 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (67 | /*op_sel*/ 0x80), (67 | /*op_sel*/ 0x80)
_v_pk_add_f16__vop3p 37, 242, 293, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 47, 242, 303, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 57, 242, 313, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 67, 242, 323, 0x0, 0x2, 0x0, 0x0
v_rcp_f16 v37, v37
v_rcp_f16 v47, v47
v_rcp_f16 v57, v57
v_rcp_f16 v67, v67
_v_rcp_f16__vop1 (37 | /*op_sel*/ 0x80), (37 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (47 | /*op_sel*/ 0x80), (47 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (57 | /*op_sel*/ 0x80), (57 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (67 | /*op_sel*/ 0x80), (67 | /*op_sel*/ 0x80)
v_pk_fma_f16 v37, v37, 2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0] neg_lo:[1,0,0] neg_hi:[1,0,0]
v_pk_fma_f16 v47, v47, 2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0] neg_lo:[1,0,0] neg_hi:[1,0,0]
v_pk_fma_f16 v57, v57, 2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0] neg_lo:[1,0,0] neg_hi:[1,0,0]
v_pk_fma_f16 v67, v67, 2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0] neg_lo:[1,0,0] neg_hi:[1,0,0]
_v_cmp_gt_f16__vop3_v_lit 106, 163, 0x38b838b8, 0x0, 0x1
v_cndmask_b16 v42, v42, v37, vcc
_v_cmp_gt_f16__vop3_v_lit 106, 164, 0x38b838b8, 0x0, 0x1
v_cndmask_b16 v52, v52, v47, vcc
_v_cmp_gt_f16__vop3_v_lit 106, 165, 0x38b838b8, 0x0, 0x1
v_cndmask_b16 v62, v62, v57, vcc
_v_cmp_gt_f16__vop3_v_lit 106, 166, 0x38b838b8, 0x0, 0x1
v_cndmask_b16 v148, v148, v67, vcc
_v_cmp_gt_f16__vop3_v_lit 106, 163, 0x38b838b8, 0x3, 0x1
_v_cndmask_b16__vop3 42, 42, 37, 106, 0xb
_v_cmp_gt_f16__vop3_v_lit 106, 164, 0x38b838b8, 0x3, 0x1
_v_cndmask_b16__vop3 52, 52, 47, 106, 0xb
_v_cmp_gt_f16__vop3_v_lit 106, 165, 0x38b838b8, 0x3, 0x1
_v_cndmask_b16__vop3 62, 62, 57, 106, 0xb
_v_cmp_gt_f16__vop3_v_lit 106, 166, 0x38b838b8, 0x3, 0x1
_v_cndmask_b16__vop3 148, 148, 67, 106, 0xb
v_bfi_b32 v163, 0x7fff7fff, v42, v163
v_bfi_b32 v164, 0x7fff7fff, v52, v164
v_bfi_b32 v165, 0x7fff7fff, v62, v165
v_bfi_b32 v166, 0x7fff7fff, v148, v166
_v_pk_mul_f16__vop3p 163, 419, 32, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 164, 420, 32, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 165, 421, 32, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 166, 422, 32, 0x0, 0x3, 0x0, 0x0
buffer_store_b16 v151, v24, s[76:79], 0 idxen
buffer_store_b16 v155, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_d16_hi_b16 v151, v24, s[76:79], 0 idxen
buffer_store_d16_hi_b16 v155, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_b16 v152, v24, s[76:79], 0 idxen
buffer_store_b16 v156, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_d16_hi_b16 v152, v24, s[76:79], 0 idxen
buffer_store_d16_hi_b16 v156, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_b16 v153, v24, s[76:79], 0 idxen
buffer_store_b16 v157, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_d16_hi_b16 v153, v24, s[76:79], 0 idxen
buffer_store_d16_hi_b16 v157, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_b16 v154, v24, s[76:79], 0 idxen
buffer_store_b16 v158, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_d16_hi_b16 v154, v24, s[76:79], 0 idxen
buffer_store_d16_hi_b16 v158, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_b16 v159, v24, s[76:79], 0 idxen
buffer_store_b16 v163, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_d16_hi_b16 v159, v24, s[76:79], 0 idxen
buffer_store_d16_hi_b16 v163, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_b16 v160, v24, s[76:79], 0 idxen
buffer_store_b16 v164, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_d16_hi_b16 v160, v24, s[76:79], 0 idxen
buffer_store_d16_hi_b16 v164, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_b16 v161, v24, s[76:79], 0 idxen
buffer_store_b16 v165, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_d16_hi_b16 v161, v24, s[76:79], 0 idxen
buffer_store_d16_hi_b16 v165, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_b16 v162, v24, s[76:79], 0 idxen
buffer_store_b16 v166, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
buffer_store_d16_hi_b16 v162, v24, s[76:79], 0 idxen
buffer_store_d16_hi_b16 v166, v25, s[76:79], 0 idxen
s_sub_u32 s75, s75, 1
s_cselect_b32 s79, 0, s79
s_add_u32 s76, s76, s80
s_addc_u32 s77, s77, 0
s_mov_b32 s86, 0x4f50
s_setpc_b64 s[88:89]
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

