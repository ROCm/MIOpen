/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

#include "float_types.h"

__kernel void UniversalTranspose(const __global _FLOAT* in,
                                 __global _FLOAT* out,
                                 ulong lens_n,
                                 ulong lens_c,
                                 ulong lens_d,
                                 ulong lens_h,
                                 ulong lens_w,
                                 ulong in_strides_n,
                                 ulong in_strides_c,
                                 ulong in_strides_d,
                                 ulong in_strides_h,
                                 ulong in_strides_w,
                                 ulong out_strides_n,
                                 ulong out_strides_c,
                                 ulong out_strides_d,
                                 ulong out_strides_h,
                                 ulong out_strides_w)
{
    const ulong local_size = get_local_size(0);
    const ulong local_id   = get_local_id(0);

    const ulong lens_wh    = lens_w * lens_h;
    const ulong lens_whd   = lens_wh * lens_d;
    const ulong lens_whdc  = lens_whd * lens_c;
    const ulong lens_whdcn = lens_whdc * lens_n;

    for(ulong id = local_id; id < lens_whdcn; id += local_size)
    {
        const ulong n = id / lens_whdc;
        const ulong c = (id / lens_whd) % lens_c;
        const ulong d = (id / lens_wh) % lens_d;
        const ulong h = (id / lens_w) % lens_h;
        const ulong w = id % lens_w;

        // clang-format off
        const ulong in_id =
            n * in_strides_n +
            c * in_strides_c +
            d * in_strides_d +
            h * in_strides_h +
            w * in_strides_w;

        const ulong out_id =
            n * out_strides_n +
            c * out_strides_c +
            d * out_strides_d +
            h * out_strides_h +
            w * out_strides_w;
        // clang-format on

        out[out_id] = in[in_id];
    }
}
