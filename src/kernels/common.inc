/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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

LAYOUT_DATA_NCHW = 0
LAYOUT_DATA_CNHW = 1
LAYOUT_DATA_NHWC = 2
LAYOUT_DATA_CHWN = 3

// acc_type and buf_type: 0 - FP64, 1 - FP32, 2 - FP16, 5 - int32, 6 - int16, 7 - int8
TYPE_FP64  = 0
TYPE_FP32  = 1
TYPE_FP16  = 2
TYPE_BFP16 = 3
TYPE_INT64 = 4
TYPE_INT32 = 5
TYPE_INT16 = 6
TYPE_INT8  = 7
TYPE_INT4  = 8

.macro log2 lg2, num, max_bits=8
	\lg2 = 0
	lg_i = \num
	.rept \max_bits
		lg_i = lg_i / 2
		.if lg_i > 0
			\lg2 = \lg2 + 1
		.endif
	.endr
.endm

.macro default symbol, value
	.ifnotdef \symbol
		\symbol = \value
	.endif
.endm

.macro static_assert fufufu
	.if !\fufufu
		.error "\fufufu is false"
		.end
	.endif
.endm

.macro swap a, b
	__tmp = \a
	\a = \b
	\b = __tmp
.endm

.macro m_bpermute vgpr, cnt, addr
	v = \vgpr
	.rept \cnt
		ds_bpermute_b32 v[v], v[\addr], v[v]
		v = v + 1
	.endr
.endm

.macro m_swizzle vgpr, cnt, pattern
	v = \vgpr
	.rept \cnt
		ds_swizzle_b32 v[v], v[v] offset:\pattern
		v = v + 1
	.endr
.endm

.if (.option.machine_version_major == 8)
	.set max_hw_vctn, 15
.elseif (.option.machine_version_major == 9)
	.set max_hw_vctn, 63
.endif
max_hw_lcnt = 15
.macro s_wait vmcnt=max_hw_vctn, lgkmcnt=max_hw_lcnt
	vm_cnt = \vmcnt
	lgkm_cnt = \lgkmcnt
	.if vm_cnt > max_hw_vctn
		vm_cnt = max_hw_vctn
	.elseif vm_cnt < 0
		vm_cnt = 0
	.endif
	.if lgkm_cnt > max_hw_lcnt
		lgkm_cnt = max_hw_lcnt
	.elseif lgkm_cnt < 0
		lgkm_cnt = 0
	.endif
	s_waitcnt vmcnt(0 + vm_cnt) & lgkmcnt(0 + lgkm_cnt)
.endm


maxU24 = 1 << 24

wave_size = 64
log2 wave_size_log2, wave_size

.macro m_buffer_load_dwordx size, dst, off, desc, soff, ioff=0
    .if \size == 1
        buffer_load_dword v[\dst], v[\off], s[\desc:\desc + 3], s[\soff] offen offset:0+\ioff
    .elseif \size == 2
        buffer_load_dwordx2 v[\dst:\dst+\size-1], v[\off], s[\desc:\desc + 3], s[\soff] offen offset:0+\ioff
    .elseif \size == 3
        buffer_load_dwordx3 v[\dst:\dst+\size-1], v[\off], s[\desc:\desc + 3], s[\soff] offen offset:0+\ioff
    .elseif \size == 4
        buffer_load_dwordx4 v[\dst:\dst+\size-1], v[\off], s[\desc:\desc + 3], s[\soff] offen offset:0+\ioff
    .endif
.endm

.macro m_buffer_load_ushort size, dst, off, desc, soff, ioff=0
    .if \size == 1
        buffer_load_ushort v[\dst], v[\off],  s[\desc:\desc + 3], s[\soff] offen offset:0+\ioff
    .endif
.endm

.macro m_buffer_store_short size, src, off, desc, soff, ioff=0
    .if \size == 1
        buffer_store_short v[\src], v[\off],  s[\desc:\desc + 3], s[\soff] offen offset:0+\ioff
    .endif
.endm

.macro u32_div numer, denom, uquo, v_tmp, s_tmp
    v_cvt_f32_u32 v[\v_tmp+0], v[\denom]
    v_rcp_f32     v[\v_tmp+0], v[\v_tmp+0]
    v_mul_f32     v[\v_tmp+0], 0x4f800000, v[\v_tmp+0]
    v_cvt_u32_f32 v[\v_tmp+0], v[\v_tmp+0]
    v_mul_lo_u32  v[\v_tmp+1], v[\denom], v[\v_tmp+0]
    v_mul_hi_u32  v[\v_tmp+2], v[\denom], v[\v_tmp+0]
   _v_sub_co_u32  v[\v_tmp+3], vcc, 0, v[\v_tmp+1]
    v_cmp_ne_i32  s[\s_tmp:\s_tmp+1], 0, v[\v_tmp+2]
    v_cndmask_b32 v[\v_tmp+1], v[\v_tmp+3], v[\v_tmp+1], s[\s_tmp:\s_tmp+1]
    v_mul_hi_u32  v[\v_tmp+1], v[\v_tmp+1], v[\v_tmp+0]
   _v_sub_co_u32  v[\v_tmp+2], vcc, v[\v_tmp+0], v[\v_tmp+1]
   _v_add_co_u32  v[\v_tmp+0], vcc, v[\v_tmp+0], v[\v_tmp+1]
    v_cndmask_b32 v[\v_tmp+0], v[\v_tmp+0], v[\v_tmp+2], s[\s_tmp:\s_tmp+1]
    v_mul_hi_u32  v[\v_tmp+0], v[\v_tmp+0], v[\numer]
    v_mul_lo_u32  v[\v_tmp+1], v[\v_tmp+0], v[\denom]
   _v_sub_co_u32  v[\v_tmp+2], vcc, v[\numer], v[\v_tmp+1]
    v_cmp_ge_u32  s[\s_tmp:\s_tmp+1], v[\numer], v[\v_tmp+1]
    v_cmp_ge_u32  s[\s_tmp+2:\s_tmp+3], v[\v_tmp+2], v[\denom]
   _v_add_co_u32  v[\v_tmp+2], vcc, 1, v[\v_tmp+0]
    s_and_b64     s[\s_tmp+2:\s_tmp+3], s[\s_tmp:\s_tmp+1], s[\s_tmp+2:\s_tmp+3]
   _v_add_co_u32  v[\v_tmp+1], vcc, -1, v[\v_tmp+0]
    v_cndmask_b32 v[\v_tmp+2], v[\v_tmp+0], v[\v_tmp+2], s[\s_tmp+2:\s_tmp+3]
    v_cndmask_b32 v[\v_tmp+2], v[\v_tmp+1], v[\v_tmp+2], s[\s_tmp:\s_tmp+1]
    v_cmp_ne_i32  vcc, 0, v[\denom]
    v_cndmask_b32 v[\uquo], -1,  v[\v_tmp+2], vcc
.endm
