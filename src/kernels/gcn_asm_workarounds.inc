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

// Wrapper macros for some instructions.
// Macros contain workarounds for some assembler bugs.
// Also these allow for unifying source text when different
// ISA versions require different assembler syntax (mostly mnemonics).

.if ((.option.machine_version_major == 8) || (.option.machine_version_major == 9))
.else
    .error "Only Gfx8 and Gfx9 ISA is supported"
    .end
.endif

.ifndef WORKAROUND_BUG_34765
    .set WORKAROUND_BUG_34765,0
.endif

// Let's use Gfx10-like naming conventions for wrapper macros.
// ADD_NC
.macro _v_add_nc_u32 dst, src0, src1, dpp=
    .if (.option.machine_version_major == 8)
        // None No-Carry instruction in Gfx8, modifies VCC.
        v_add_u32 \dst, vcc, \src0, \src1 \dpp
    .else
        v_add_u32 \dst, \src0, \src1 \dpp
    .endif
.endm

// ADD_CO (gfx8 add)
.macro _v_add_co_u32 dst, co, src0, src1, dpp=
    .if ((.option.machine_version_major == 8) || ((.option.machine_version_major == 9) && (WORKAROUND_BUG_34765 == 1)))
        v_add_u32 \dst, \co, \src0, \src1 \dpp
    .else
        v_add_co_u32 \dst, \co, \src0, \src1 \dpp
    .endif
.endm

// ADD_CO_CI (gfx8 addc)
.macro _v_add_co_ci_u32 dst, co, src0, src1, ci, dpp=
    .if ((.option.machine_version_major == 8) || ((.option.machine_version_major == 9) && (WORKAROUND_BUG_34765 == 1)))
		v_addc_u32 \dst, \co, \src0, \src1, \ci \dpp
    .else
		v_addc_co_u32 \dst, \co, \src0, \src1, \ci \dpp
    .endif
.endm

// SUB_CO (gfx8 sub)
.macro _v_sub_co_u32 dst, co, src0, src1, dpp=
    .if ((.option.machine_version_major == 8) || ((.option.machine_version_major == 9) && (WORKAROUND_BUG_34765 == 1)))
        v_sub_u32 \dst, \co, \src0, \src1 \dpp
    .else
        v_sub_co_u32 \dst, \co, \src0, \src1 \dpp
    .endif
.endm

// SUBREV_CO (gfx8 subrev)
.macro _v_subrev_co_u32 dst, co, src0, src1, dpp=
    .if ((.option.machine_version_major == 8) || ((.option.machine_version_major == 9) && (WORKAROUND_BUG_34765 == 1)))
        v_subrev_u32 \dst, \co, \src0, \src1 \dpp
    .else
        v_subrev_co_u32 \dst, \co, \src0, \src1 \dpp
    .endif
.endm

