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

.set __auto_gpr_count_guard, 1

.set .WAVE_SIZE, 64
.set .WAVE_SIZE_LOG2, 6
.set .MAX_VGPRS, 256
.set .MAX_SGPRS, 102
.set .MAX_LDS, 65536

.macro .GPR_ALLOC_BEGIN
    .set .AVAILABLE_VGPRS, .MAX_VGPRS
    .set .AVAILABLE_SGPRS, .MAX_SGPRS
	.set .AVAILABLE_LDS, .MAX_LDS
	.set .SGPR_NEXT_FREE, 0
	.set .VGPR_NEXT_FREE, 0
	.set .LDS_NEXT_FREE, 0
	.set .AUTO_VGPR_COUNT, 0
	.set .AUTO_SGPR_COUNT, 0
	.set .AUTO_LDS_BYTE_SIZE, 0
	.set .AUTO_VGPR_GRANULATED_COUNT, 0
	.set .AUTO_SGPR_GRANULATED_COUNT, 0
	.set __sgpr_reserve_vcc, 0
	.set __sgpr_reserve_xnack, 0
	.set __sgpr_reserve_flatscr, 0
	.set __auto_gpr_count_guard, 0
	.set __max_waves_limit, 10
	.set __min_waves_limit, 1
.endm

.macro .CHECK_SGPR_ALLOCATION gprs_to_allocate=0
	.if (.SGPR_NEXT_FREE + \gprs_to_allocate) >= .AVAILABLE_SGPRS
		.error "Error: out of free sgprs"
        .end
	.endif
.endm

.macro .CHECK_VGPR_ALLOCATION gprs_to_allocate=0
	.if (.VGPR_NEXT_FREE + \gprs_to_allocate) >= .AVAILABLE_VGPRS
		.error "Error: out of free vgprs"
        .end
	.endif
.endm

.macro .CHECK_LDS_ALLOCATION bytes_to_allocate=0
	.if (.LDS_NEXT_FREE + \bytes_to_allocate) >= .AVAILABLE_LDS
		.error "Error: out of free lds"
        .end
	.endif
.endm

.macro .GPRS_FOR_WAVE_LIMIT waves_per_simd, sgprs, vgprs
    .if \waves_per_simd == 10
	    \sgprs = 80
		\vgprs = 24
	.elseif \waves_per_simd == 9
	    \sgprs = 96
		\vgprs = 28
	.elseif \waves_per_simd == 8
	    \sgprs = 96
		\vgprs = 32
	.elseif \waves_per_simd == 7
	    \sgprs = 102
		\vgprs = 36
	.elseif \waves_per_simd == 6
	    \sgprs = 102
		\vgprs = 40
	.elseif \waves_per_simd == 5
	    \sgprs = 102
		\vgprs = 48
	.elseif \waves_per_simd == 4
	    \sgprs = 102
		\vgprs = 64
	.elseif \waves_per_simd == 3
	    \sgprs = 102
		\vgprs = 84
	.elseif \waves_per_simd == 2
	    \sgprs = 102
		\vgprs = 128
	.else
	    \sgprs = 102
		\vgprs = 256
	.endif
.endm

.macro .SET_MIN_WAVES_LIMIT waves_per_simd
    .if \waves_per_simd > 10
		.error "Error: max 10 waves per simd is available"
	.endif
	.GPRS_FOR_WAVE_LIMIT \waves_per_simd, .AVAILABLE_SGPRS, .AVAILABLE_VGPRS
	.CHECK_SGPR_ALLOCATION
	.CHECK_VGPR_ALLOCATION
	__min_waves_limit = \waves_per_simd
	.if __min_waves_limit > __max_waves_limit
	    .error "Error: __min_waves_limit > __max_waves_limit"
	.endif
.endm

.macro .SET_MAX_WAVES_LIMIT waves_per_simd
    .if \waves_per_simd < 1
		.error "Error: waves per simd should be > 0"
	.endif
	__max_waves_limit = \waves_per_simd
	.if __min_waves_limit > __max_waves_limit
	    .error "Error: __min_waves_limit > __max_waves_limit"
	.endif
.endm


.macro .GPR_ALLOC_END
    .if __auto_gpr_count_guard == 1
	    .error "Error: unpaired .GPR_ALLOC_END. Please invoke .GPR_ALLOC_BEGIN before each kernel."
	.endif
	.CHECK_SGPR_ALLOCATION
	.CHECK_VGPR_ALLOCATION
	.CHECK_LDS_ALLOCATION
	__sgpr_additional_count = 2 * (__sgpr_reserve_flatscr + __sgpr_reserve_xnack + __sgpr_reserve_vcc)
	.GPRS_FOR_WAVE_LIMIT __max_waves_limit, .AUTO_SGPR_COUNT, .AUTO_VGPR_COUNT
	.if .AUTO_VGPR_COUNT < .VGPR_NEXT_FREE
	    .AUTO_VGPR_COUNT = .VGPR_NEXT_FREE
	.endif
	.if .AUTO_SGPR_COUNT < (.SGPR_NEXT_FREE + __sgpr_additional_count)
	    .AUTO_SGPR_COUNT = (.SGPR_NEXT_FREE + __sgpr_additional_count)
	.endif
	.AUTO_VGPR_GRANULATED_COUNT = (.AUTO_VGPR_COUNT - 1)/4
	.AUTO_SGPR_GRANULATED_COUNT = (.AUTO_SGPR_COUNT - 1)/8
	.AUTO_LDS_BYTE_SIZE = .LDS_NEXT_FREE
    __auto_gpr_count_guard = 1
.endm

.macro .VGPR_ALLOC_FROM __vgpr_alloc_from
    .set .VGPR_NEXT_FREE, \__vgpr_alloc_from
.endm

.macro .SGPR_ALLOC_FROM __sgpr_alloc_from
    .set .SGPR_NEXT_FREE, \__sgpr_alloc_from
.endm

.macro .LDS_ALLOC_FROM __lds_alloc_from
	.set .LDS_NEXT_FREE, \__lds_alloc_from
.endm

.macro .SGPR_RESERVE_FLATSCR
    .set __sgpr_reserve_flatscr, 1
.endm

.macro .SGPR_RESERVE_XNACK
    .set __sgpr_reserve_xnack, 1
.endm

.macro .SGPR_RESERVE_VCC
    .set __sgpr_reserve_vcc, 1
.endm

.macro .VGPR_ALLOC __vgpr_number_symbolic, __vgpr_numregs=1
    .CHECK_VGPR_ALLOCATION \__vgpr_numregs
    .set \__vgpr_number_symbolic, .VGPR_NEXT_FREE
    .set .VGPR_NEXT_FREE, .VGPR_NEXT_FREE + \__vgpr_numregs
.endm

.macro .SGPR_ALLOC __sgpr_number_symbolic, __sgpr_numregs=1, __sgpr_alignment=0
    .CHECK_SGPR_ALLOCATION \__sgpr_numregs
	.if \__sgpr_alignment > 0
		.set __sgpr_effective_alignment, \__sgpr_alignment
	.elseif \__sgpr_numregs > 4
		.set __sgpr_effective_alignment, 4
	.else
		.set __sgpr_effective_alignment, \__sgpr_numregs
	.endif
    .if .SGPR_NEXT_FREE % __sgpr_effective_alignment != 0
		.error "Error: unaligned register"
    .endif
    .set \__sgpr_number_symbolic, .SGPR_NEXT_FREE
    .set .SGPR_NEXT_FREE, .SGPR_NEXT_FREE + \__sgpr_numregs
.endm

.macro .LDS_ALLOC __lds_ptr_name, __bytes_to_allocate, __alignment_size=1
	.if .LDS_NEXT_FREE % \__alignment_size != 0
		.LDS_ALLOC_FROM ((.LDS_NEXT_FREE / \__alignment_size) + 1) * \__alignment_size
	.endif
	.CHECK_LDS_ALLOCATION \__bytes_to_allocate
	.set \__lds_ptr_name, .LDS_NEXT_FREE
	.set .LDS_NEXT_FREE, .LDS_NEXT_FREE + \__bytes_to_allocate
.endm

.macro .SGPR_ALLOC_ONCE __sgpr_symbolic, __sgpr_numregs=1, __sgpr_alignment=0
	.ifndef __guard_sgpr_\__sgpr_symbolic
		__guard_sgpr_\__sgpr_symbolic = 0
	.endif
	.if __guard_sgpr_\__sgpr_symbolic == 0
		__guard_sgpr_\__sgpr_symbolic = 1
		.SGPR_ALLOC \__sgpr_symbolic, \__sgpr_numregs, \__sgpr_alignment
	.endif
.endm

.macro .GPR_INVALIDATE __gpr_symbolic
	.set \__gpr_symbolic, 0x7fffffff /* invalidate (intentionally to the middle of the int range) */
.endm

.macro .GPR_REUSE __gpr_number_symbolic_old, __gpr_number_symbolic_new
    .set \__gpr_number_symbolic_new, \__gpr_number_symbolic_old
    .GPR_INVALIDATE \__gpr_number_symbolic_old
.endm
