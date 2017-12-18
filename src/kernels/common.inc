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

