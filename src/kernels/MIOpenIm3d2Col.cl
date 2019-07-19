/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#ifndef MIOPEN_USE_FP32
#define MIOPEN_USE_FP32 0
#endif

#ifndef MIOPEN_USE_FP16
#define MIOPEN_USE_FP16 0
#endif

#ifndef MIOPEN_USE_BFP16
#define MIOPEN_USE_BFP16 0
#endif

#ifndef MIOPEN_USE_INT8
#define MIOPEN_USE_INT8 0
#endif

#ifndef MIOPEN_USE_INT8x4
#define MIOPEN_USE_INT8x4 0
#endif

#ifndef MIOPEN_USE_INT32
#define MIOPEN_USE_INT32 0
#endif

#if MIOPEN_USE_INT8
typedef char data_t;
#elif MIOPEN_USE_INT8x4
typedef uint data_t;
#elif MIOPEN_USE_INT32
typedef int data_t;
#elif(MIOPEN_USE_FP16 || MIOPEN_USE_BFP16)
// As the half type degrades the performance, use short instead of half in the
// im2col, which has no match op. May change back to half when compile can
// deliver equal performance as short
typedef short data_t;
#elif MIOPEN_USE_FP32
typedef float data_t;
#endif

kernel void Im3d2Col(global data_t* const __restrict im,
                     const unsigned im_offset,
                     const unsigned im_c_size,
                     const unsigned im_d_size,
                     const unsigned im_h_size,
                     const unsigned im_w_size,
                     const unsigned wei_d_size,
                     const unsigned wei_h_size,
                     const unsigned wei_w_size,
                     const unsigned out_d_size,
                     const unsigned out_h_size,
                     const unsigned out_w_size,
                     const unsigned pad_d_size,
                     const unsigned pad_h_size,
                     const unsigned pad_w_size,
                     const unsigned stride_d_size,
                     const unsigned stride_h_size,
                     const unsigned stride_w_size,
                     const unsigned dilation_d_size,
                     const unsigned dilation_h_size,
                     const unsigned dilation_w_size,
                     global data_t* __restrict col)
{
    unsigned col_size =
        out_d_size * out_h_size * out_w_size * wei_d_size * wei_h_size * wei_w_size * im_c_size;

    for(unsigned tid = get_global_id(0); tid < col_size; tid += get_global_size(0))
    {
        // "col" matrix row and colume id
        unsigned col_i = tid / (out_d_size * out_h_size * out_w_size);
        unsigned col_j = tid - col_i * (out_d_size * out_h_size * out_w_size);

        // output tensor out_d, out_h, out_w id
        unsigned out_d = col_j / (out_h_size * out_w_size);
        unsigned tmp   = col_j - out_d * (out_h_size * out_w_size);
        unsigned out_h = tmp / out_w_size;
        unsigned out_w = tmp - out_h * out_w_size;

        // weight tensor wei_c, wei_d, wei_h, wei_d id
        unsigned wei_c = col_i / (wei_d_size * wei_h_size * wei_w_size);
        tmp            = col_i - wei_c * (wei_d_size * wei_h_size * wei_w_size);
        unsigned wei_d = tmp / (wei_h_size * wei_w_size);
        tmp -= wei_d * (wei_h_size * wei_w_size);
        unsigned wei_h = tmp / wei_w_size;
        unsigned wei_w = tmp - wei_h * wei_w_size;

        // input tensor im_d, im_h, im_w id
        int im_d = (int)(stride_d_size * out_d + dilation_d_size * wei_d) - (int)(pad_d_size);
        int im_h = (int)(stride_h_size * out_h + dilation_h_size * wei_h) - (int)(pad_h_size);
        int im_w = (int)(stride_w_size * out_w + dilation_w_size * wei_w) - (int)(pad_w_size);

        data_t value = (im_d >= 0 && im_d < im_d_size && im_h >= 0 && im_h < im_h_size &&
                        im_w >= 0 && im_w < im_w_size)
                           ? im[im_offset + wei_c * (im_d_size * im_h_size * im_w_size) +
                                im_d * (im_h_size * im_w_size) + im_h * im_w_size + im_w]
                           : 0;

        col[tid] = value;
    }
}
