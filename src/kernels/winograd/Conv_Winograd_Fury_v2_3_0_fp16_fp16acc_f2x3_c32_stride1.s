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
.include "Conv_Winograd_Fury_v2_3_0_metadata.inc"

.if (.amdgcn.gfx_generation_number == 11)
    .if ((.amdgcn.gfx_generation_minor == 0 && (.amdgcn.gfx_generation_stepping == 0 || .amdgcn.gfx_generation_stepping == 1)) || (.amdgcn.gfx_generation_minor == 5 && .amdgcn.gfx_generation_stepping == 1))
        // gfx1100, gfx1101, gfx1151
        KERNEL_PROLOG _1536vgprs_fp16_fp16acc_f2x3_c32_stride1
        .include "Conv_Winograd_Fury_v2_3_0_gfx11_1536vgprs_fp16_fp16acc_f2x3_c32_stride1.inc"
        KERNEL_EPILOG _1536vgprs_fp16_fp16acc_f2x3_c32_stride1
    .else
        // gfx1102, gfx1103, gfx1150
        .error "Unsupported gpu"
    .endif
.else
    .error "Unsupported gfx generation"
.endif
