/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
.include "Conv_Winograd_v30_2_6_metadata.inc"

.if (.amdgcn.gfx_generation_number == 9)
    KERNEL_PROLOG fp16_dot2_edc_f3x2_dilation2

    .include "Conv_Winograd_v30_2_6_gfx9_fp16_dot2_edc_f3x2_dilation2.inc"

    KERNEL_EPILOG fp16_dot2_edc_f3x2_dilation2
.elseif (.amdgcn.gfx_generation_number == 10)
    KERNEL_PROLOG fp16_dot2_f3x2_dilation2

    .include "Conv_Winograd_v30_2_6_gfx10_fp16_dot2_f3x2_dilation2.inc"

    KERNEL_EPILOG fp16_dot2_f3x2_dilation2
.else
    .error "Unsupported gfx generation"
.endif
