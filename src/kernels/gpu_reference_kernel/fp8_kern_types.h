/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
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
#pragma once

#define CAT_I(a, b) a##b
#define CAT(a, b) CAT_I(a, b)

#ifndef INPUT_TYPE
#define INPUT_TYPE half
#endif

#ifndef OUTPUT_TYPE
#define OUTPUT_TYPE half
#endif

#ifndef WEIGHTS_TYPE
#define WEIGHTS_TYPE half
#endif

#ifndef INPUT_CAST_TYPE
#define INPUT_CAST_TYPE float8_fnuz
#endif

#ifndef WEIGHTS_CAST_TYPE
#define WEIGHTS_CAST_TYPE float8_fnuz
#endif

#ifndef OUTPUT_CAST_TYPE
#define OUTPUT_CAST_TYPE float8_fnuz
#endif

#ifndef ACCUMULATOR_TYPE
#define ACCUMULATOR_TYPE double
#endif

#define KERNEL_NAME_SUFFIX CAT(CAT(INPUT_TYPE, _), CAT(CAT(WEIGHTS_TYPE, _), OUTPUT_TYPE))

#define FWD_KERNEL_NAME CAT(naive_conv_nonpacked_fwd_nchw_, KERNEL_NAME_SUFFIX)
#define BWD_KERNEL_NAME CAT(naive_conv_nonpacked_bwd_nchw_, KERNEL_NAME_SUFFIX)
#define WRW_KERNEL_NAME CAT(naive_conv_nonpacked_wrw_nchw_, KERNEL_NAME_SUFFIX)
