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

#if MIOPEN_USE_FP16 == 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define _FLOAT half
#ifndef HALF_MAX
#define MAX_VAL 65504 /* max value */
#else
#define MAX_VAL HALF_MAX
#endif
#endif
#if MIOPEN_USE_FP32 == 1
#define _FLOAT float
#ifndef FLT_MAX
#define MAX_VAL 3.402823466e+38F /* max value */
#else
#define MAX_VAL FLT_MAX
#endif
#endif

/* Only works for NCHW
 * bitmap tracks which dims are the same between 'a' and 'c'.
 * Example: 0, 1, 1, 0 means that C and H dims are the same and the rest are ones
 * bitmap dims with 0 contribute to the work_per_wg,
 * whereas dims with 1 contribute to the #workgroups (gid)
 * work_per_wg = product of dims with 0s (dims of 'c tensor') and
 * num_wg = product of dims with 1s (dims of 'a')
 * Bitmap for fwd_bias looks like 0, 1, 0, 0
 */

#ifndef MIOPEN_TENSOR_OP
#define MIOPEN_TENSOR_OP miopenMul
#endif

#define UNUSED __attribute__((__unused__))

MIOPEN_TYPE miopenAdd(MIOPEN_TYPE a, MIOPEN_TYPE b) { return a + b; }

MIOPEN_TYPE miopenMul(MIOPEN_TYPE a, MIOPEN_TYPE b) { return a * b; }

MIOPEN_TYPE miopenMax(MIOPEN_TYPE a, MIOPEN_TYPE b) { return ((a > b) ? a : b); }

MIOPEN_TYPE miopenMin(MIOPEN_TYPE a, MIOPEN_TYPE b) { return ((a < b) ? a : b); }

#ifdef USE_FWD_BIAS

__kernel void OpTensorFwdBias(global MIOPEN_TYPE* a,
                              global MIOPEN_TYPE* b,
                              const int b_c,
                              global MIOPEN_TYPE* c,
                              const int c_n,
                              const int c_nstride,
                              const int c_cstride,
                              const int work_per_wg,
                              const MIOPEN_TYPE alpha0,
                              const MIOPEN_TYPE alpha1,
                              const MIOPEN_TYPE beta,
                              const long Aoffset,
                              const long Boffset,
                              const long Coffset,
                              const int num_wg,
                              const int incr_wg)
{
    global MIOPEN_TYPE* a_off = a + Aoffset;
    global MIOPEN_TYPE* b_off = b + Boffset;
    global MIOPEN_TYPE* c_off = c + Coffset;

    int gid = get_group_id(0);

    // num_wg: the number of workgroups should be launched
    // MAX_NUM_WG: the maximum number of workgroups actually launched

    if(beta == (MIOPEN_TYPE)0)
    {
        for(; gid < num_wg; gid += MAX_NUM_WG)
        {

            int lid = get_local_id(0);

            int o_c             = incr_wg == 1 ? (gid % b_c) : gid;
            MIOPEN_TYPE operand = b_off[o_c] * alpha1;

            // each workgroup computes N*H*W for each C (bias-term)
            // number of workgroups = c_c (b_c)
            while(lid < work_per_wg)
            {
                int o_hw     = incr_wg == 0 ? (lid % (work_per_wg / c_n)) : lid;
                int o_n      = incr_wg == 0 ? (lid / (work_per_wg / c_n)) : (gid / b_c);
                int index    = o_n * c_nstride + o_c * c_cstride + o_hw;
                c_off[index] = MIOPEN_TENSOR_OP(a_off[index] * alpha0, operand);

                lid += get_local_size(0);
            }
        }
    }
    else
    {
        for(; gid < num_wg; gid += MAX_NUM_WG)
        {

            int lid = get_local_id(0);

            int o_c             = incr_wg == 1 ? (gid % b_c) : gid;
            MIOPEN_TYPE operand = b_off[o_c] * alpha1;

            // each workgroup computes N*H*W for each C (bias-term)
            // number of workgroups = c_c (b_c)
            while(lid < work_per_wg)
            {
                int o_hw  = incr_wg == 0 ? (lid % (work_per_wg / c_n)) : lid;
                int o_n   = incr_wg == 0 ? (lid / (work_per_wg / c_n)) : (gid / b_c);
                int index = o_n * c_nstride + o_c * c_cstride + o_hw;
                c_off[index] =
                    MIOPEN_TENSOR_OP(a_off[index] * alpha0, operand) + beta * c_off[index];

                lid += get_local_size(0);
            }
        }
    }
}

#endif

#ifdef USE_FWD_BIAS_GENERIC

__kernel void OpTensorFwdBiasGeneric(global MIOPEN_TYPE* a,
                                     const int a_nstride,
                                     const int a_cstride,
                                     const int a_hstride,
                                     global MIOPEN_TYPE* b,
                                     const int b_c,
                                     const int b_cstride,
                                     global MIOPEN_TYPE* c,
                                     const int c_n,
                                     const int c_w,
                                     const int c_nstride,
                                     const int c_cstride,
                                     const int c_hstride,
                                     const MIOPEN_TYPE alpha0,
                                     const MIOPEN_TYPE alpha1,
                                     const MIOPEN_TYPE beta,
                                     const int work_per_wg,
                                     const long Aoffset,
                                     const long Boffset,
                                     const long Coffset,
                                     const int num_wg,
                                     const int incr_wg)
{
    int gid = get_group_id(0);

    global MIOPEN_TYPE* a_off = a + Aoffset;
    global MIOPEN_TYPE* b_off = b + Boffset;
    global MIOPEN_TYPE* c_off = c + Coffset;

    // num_wg: the number of workgroups should be launched
    // MAX_NUM_WG: the maximum number of workgroups actually launched
    if(beta == (MIOPEN_TYPE)0)
    {
        for(; gid < num_wg; gid += MAX_NUM_WG)
        {
            int lid = get_local_id(0);

            // each workgroup computes N*H*W for each C (bias-term)
            // number of workgroups = c_c (b_c)
            int o_c             = (incr_wg == 1) ? (gid % b_c) : gid;
            MIOPEN_TYPE operand = b_off[o_c * b_cstride] * alpha1;

            while(lid < work_per_wg)
            {
                int o_n       = (incr_wg == 1) ? (gid / b_c) : (lid % c_n);
                int o_h       = (incr_wg == 1) ? (lid / c_w) : ((lid / c_n) / c_w);
                int o_w       = (incr_wg == 1) ? (lid % c_w) : ((lid / c_n) % c_w);
                int aindex    = o_n * a_nstride + o_c * a_cstride + o_h * a_hstride + o_w;
                int cindex    = o_n * c_nstride + o_c * c_cstride + o_h * c_hstride + o_w;
                c_off[cindex] = MIOPEN_TENSOR_OP(a_off[aindex] * alpha0, operand);

                lid += get_local_size(0);
            }
        }
    }
    else
    {
        for(; gid < num_wg; gid += MAX_NUM_WG)
        {
            int lid = get_local_id(0);

            // each workgroup computes N*H*W for each C (bias-term)
            // number of workgroups = c_c (b_c)
            int o_c             = (incr_wg == 1) ? (gid % b_c) : gid;
            MIOPEN_TYPE operand = b_off[o_c * b_cstride] * alpha1;

            while(lid < work_per_wg)
            {
                int o_n    = (incr_wg == 1) ? (gid / b_c) : (lid % c_n);
                int o_h    = (incr_wg == 1) ? (lid / c_w) : ((lid / c_n) / c_w);
                int o_w    = (incr_wg == 1) ? (lid % c_w) : ((lid / c_n) % c_w);
                int aindex = o_n * a_nstride + o_c * a_cstride + o_h * a_hstride + o_w;
                int cindex = o_n * c_nstride + o_c * c_cstride + o_h * c_hstride + o_w;
                c_off[cindex] =
                    MIOPEN_TENSOR_OP(a_off[aindex] * alpha0, operand) + beta * c_off[cindex];

                lid += get_local_size(0);
            }
        }
    }
}

#endif

// DLOWELL : cutting out this section
#ifdef USE_LEADING_ONES

__kernel void OpTensorLeadingOnes(global MIOPEN_TYPE* a,
                                  global MIOPEN_TYPE* b,
                                  global MIOPEN_TYPE* c,
                                  const int c_c,
                                  const int c_h,
                                  const int c_w,
                                  const int c_nstride,
                                  const int c_cstride,
                                  const int work_per_wg,
                                  const MIOPEN_TYPE alpha0,
                                  const MIOPEN_TYPE alpha1,
                                  const MIOPEN_TYPE beta,
                                  const long Aoffset,
                                  const long Boffset,
                                  const long Coffset,
                                  const int num_wg,
                                  const unsigned int bitmap)
{

    /* Special case for leading ones where the total no. of threads is the
     * inner_product of the tensor dims.  Each thread just updates one value
     */

    global MIOPEN_TYPE* a_off = a + Aoffset;
    global MIOPEN_TYPE* b_off = b + Boffset;
    global MIOPEN_TYPE* c_off = c + Coffset;

    int gid = (bitmap == 0xF) ? get_global_id(0) : get_group_id(0);
    // num_wg: the number of workgroups should be launched
    // MAX_NUM_WG: the maximum number of workgroups actually launched
    if(beta == (MIOPEN_TYPE)0)
    {
        for(; gid < num_wg; gid += MAX_NUM_WG)
        {

            int lid             = (bitmap == 0xF) ? 0 : get_local_id(0);
            int lcl_sz          = (bitmap == 0xF) ? work_per_wg : get_local_size(0);
            MIOPEN_TYPE operand = b_off[gid] * alpha1;

            int o_w = (bitmap & (1 << 0)) ? (gid % c_w) : 0;
            int o_h = (bitmap & (1 << 1)) ? ((gid / ((bitmap & (1 << 0)) ? c_w : 1)) % c_h) : 0;
            int o_c =
                (bitmap & (1 << 2))
                    ? ((gid / (((bitmap & (1 << 0)) ? c_w : 1) * ((bitmap & (1 << 1)) ? c_h : 1))) %
                       c_c)
                    : 0;
            int o_n = gid / (((bitmap & (1 << 0)) ? c_w : 1) * ((bitmap & (1 << 1)) ? c_h : 1) *
                             ((bitmap & (1 << 2)) ? c_c : 1));

            while(lid < work_per_wg)
            {
                int index    = o_n * c_nstride + o_c * c_cstride + o_h * c_w + o_w + lid;
                c_off[index] = MIOPEN_TENSOR_OP(a_off[index] * alpha0, operand);
                lid += lcl_sz;
            }
        }
    }
    else
    {
        for(; gid < num_wg; gid += MAX_NUM_WG)
        {

            int lid             = (bitmap == 0xF) ? 0 : get_local_id(0);
            int lcl_sz          = (bitmap == 0xF) ? work_per_wg : get_local_size(0);
            MIOPEN_TYPE operand = b_off[gid] * alpha1;

            int o_w = (bitmap & (1 << 0)) ? (gid % c_w) : 0;
            int o_h = (bitmap & (1 << 1)) ? ((gid / ((bitmap & (1 << 0)) ? c_w : 1)) % c_h) : 0;
            int o_c =
                (bitmap & (1 << 2))
                    ? ((gid / (((bitmap & (1 << 0)) ? c_w : 1) * ((bitmap & (1 << 1)) ? c_h : 1))) %
                       c_c)
                    : 0;
            int o_n = gid / (((bitmap & (1 << 0)) ? c_w : 1) * ((bitmap & (1 << 1)) ? c_h : 1) *
                             ((bitmap & (1 << 2)) ? c_c : 1));

            while(lid < work_per_wg)
            {
                int index = o_n * c_nstride + o_c * c_cstride + o_h * c_w + o_w + lid;
                c_off[index] =
                    MIOPEN_TENSOR_OP(a_off[index] * alpha0, operand) + beta * c_off[index];
                lid += lcl_sz;
            }
        }
    }
}

#endif

// DLOWELL : cutting out this section
#ifdef USE_LEADING_ONES_GENERIC

__kernel void OpTensorLeadingOnesGeneric(global MIOPEN_TYPE* a,
                                         const int a_nstride,
                                         const int a_cstride,
                                         const int a_hstride,
                                         global MIOPEN_TYPE* b,
                                         const int b_nstride,
                                         const int b_cstride,
                                         const int b_hstride,
                                         global MIOPEN_TYPE* c,
                                         const int c_c,
                                         const int c_h,
                                         const int c_w,
                                         const int c_nstride,
                                         const int c_cstride,
                                         const int c_hstride,
                                         const MIOPEN_TYPE alpha0,
                                         const MIOPEN_TYPE alpha1,
                                         const MIOPEN_TYPE beta,
                                         const int work_per_wg,
                                         const long Aoffset,
                                         const long Boffset,
                                         const long Coffset,
                                         const int num_wg,
                                         const unsigned int bitmap)
{

    /* Special case for leading ones where the total no. of threads is the
     * inner_product of the tensor dims.  Each thread just updates one value
     */
    global MIOPEN_TYPE* a_off = a + Aoffset;
    global MIOPEN_TYPE* b_off = b + Boffset;
    global MIOPEN_TYPE* c_off = c + Coffset;

    int gid = (bitmap == 0xF) ? get_global_id(0) : get_group_id(0);

    // num_wg: the number of workgroups should be launched
    // MAX_NUM_WG: the maximum number of workgroups actually launched
    if(beta == (MIOPEN_TYPE)0)
    {
        for(; gid < num_wg; gid += MAX_NUM_WG)
        {
            int lid    = (bitmap == 0xF) ? 0 : get_local_id(0);
            int lcl_sz = (bitmap == 0xF) ? work_per_wg : get_local_size(0);

            int o_w = (bitmap & (1 << 0)) ? (gid % c_w) : 0;
            int o_h = (bitmap & (1 << 1)) ? ((gid / ((bitmap & (1 << 0)) ? c_w : 1)) % c_h) : 0;
            int o_c =
                (bitmap & (1 << 2))
                    ? ((gid / (((bitmap & (1 << 0)) ? c_w : 1) * ((bitmap & (1 << 1)) ? c_h : 1))) %
                       c_c)
                    : 0;
            int o_n = gid / (((bitmap & (1 << 0)) ? c_w : 1) * ((bitmap & (1 << 1)) ? c_h : 1) *
                             ((bitmap & (1 << 2)) ? c_c : 1));

            int bindex          = o_n * b_nstride + o_c * b_cstride + o_h * b_hstride + o_w;
            MIOPEN_TYPE operand = b_off[bindex] * alpha1;

            while(lid < work_per_wg)
            {
                o_c           = (bitmap & (1 << 2)) ? o_c : (lid % c_c);
                o_h           = (bitmap & (1 << 1))
                                    ? o_h
                                    : ((bitmap & (1 << 2)) ? (lid / c_w) : ((lid / c_c) % c_h));
                o_w           = (bitmap & (1 << 0))
                                    ? o_w
                                    : ((bitmap & (1 << 1))
                                           ? lid
                                           : ((bitmap & (1 << 2)) ? (lid % c_w) : ((lid / c_c) / c_h)));
                int aindex    = o_n * a_nstride + o_c * a_cstride + o_h * a_hstride + o_w;
                int cindex    = o_n * c_nstride + o_c * c_cstride + o_h * c_hstride + o_w;
                c_off[cindex] = MIOPEN_TENSOR_OP(a_off[aindex] * alpha0, operand);

                lid += lcl_sz;
            }
        }
    }
    else
    {
        for(; gid < num_wg; gid += MAX_NUM_WG)
        {
            int lid    = (bitmap == 0xF) ? 0 : get_local_id(0);
            int lcl_sz = (bitmap == 0xF) ? work_per_wg : get_local_size(0);

            int o_w = (bitmap & (1 << 0)) ? (gid % c_w) : 0;
            int o_h = (bitmap & (1 << 1)) ? ((gid / ((bitmap & (1 << 0)) ? c_w : 1)) % c_h) : 0;
            int o_c =
                (bitmap & (1 << 2))
                    ? ((gid / (((bitmap & (1 << 0)) ? c_w : 1) * ((bitmap & (1 << 1)) ? c_h : 1))) %
                       c_c)
                    : 0;
            int o_n = gid / (((bitmap & (1 << 0)) ? c_w : 1) * ((bitmap & (1 << 1)) ? c_h : 1) *
                             ((bitmap & (1 << 2)) ? c_c : 1));

            int bindex          = o_n * b_nstride + o_c * b_cstride + o_h * b_hstride + o_w;
            MIOPEN_TYPE operand = b_off[bindex] * alpha1;

            while(lid < work_per_wg)
            {
                o_c        = (bitmap & (1 << 2)) ? o_c : (lid % c_c);
                o_h        = (bitmap & (1 << 1))
                                 ? o_h
                                 : ((bitmap & (1 << 2)) ? (lid / c_w) : ((lid / c_c) % c_h));
                o_w        = (bitmap & (1 << 0))
                                 ? o_w
                                 : ((bitmap & (1 << 1))
                                        ? lid
                                        : ((bitmap & (1 << 2)) ? (lid % c_w) : ((lid / c_c) / c_h)));
                int aindex = o_n * a_nstride + o_c * a_cstride + o_h * a_hstride + o_w;
                int cindex = o_n * c_nstride + o_c * c_cstride + o_h * c_hstride + o_w;
                c_off[cindex] =
                    MIOPEN_TENSOR_OP(a_off[aindex] * alpha0, operand) + beta * c_off[cindex];

                lid += lcl_sz;
            }
        }
    }
}

#endif

#ifdef USE_4D_TENSOR_GENERIC

__kernel void Op4dTensorGeneric(global MIOPEN_TYPE* a,
                                const int a_nstride,
                                const int a_cstride,
                                const int a_hstride,
                                global MIOPEN_TYPE* b,
                                const int b_c,
                                const int b_h,
                                const int b_w,
                                const int b_nstride,
                                const int b_cstride,
                                const int b_hstride,
                                global MIOPEN_TYPE* c,
                                const int c_c,
                                const int c_h,
                                const int c_w,
                                const int c_nstride,
                                const int c_cstride,
                                const int c_hstride,
                                const MIOPEN_TYPE alpha0,
                                const MIOPEN_TYPE alpha1,
                                const MIOPEN_TYPE beta,
                                const unsigned int bitmap,
                                const int work_per_wg,
                                const long Aoffset,
                                const long Boffset,
                                const long Coffset,
                                const int num_wg)
{
    int gid = get_group_id(0);

    global MIOPEN_TYPE* a_off = a + Aoffset;
    global MIOPEN_TYPE* b_off = b + Boffset;
    global MIOPEN_TYPE* c_off = c + Coffset;

    // MIOPEN_TYPE operand = b[gid + Boffset];
    // num_wg: the number of workgroups should be launched
    // MAX_NUM_WG: the maximum number of workgroups actually launched
    if(beta == (MIOPEN_TYPE)0)
    {
        for(; gid < num_wg; gid += MAX_NUM_WG)
        {
            int lid = get_local_id(0);

            int o_h_div = bitmap & (1 << 0) ? 1 : c_w;
            int o_c_div = o_h_div * (bitmap & (1 << 1) ? 1 : c_h);
            int o_n_div = o_c_div * (bitmap & (1 << 2) ? 1 : c_c);

            int o_w_gid_off = gid % b_w;
            int o_h_gid_off = (gid / b_w) % b_h;
            int o_c_gid_off = (gid / b_w / b_h) % b_c;
            int o_n_gid_off = (gid / b_w / b_h) / b_c;

            int bindex = o_n_gid_off * b_nstride + o_c_gid_off * b_cstride +
                         o_h_gid_off * b_hstride + o_w_gid_off;
            MIOPEN_TYPE operand = b_off[bindex] * alpha1;

            while(lid < work_per_wg)
            {
                int o_w = (bitmap & (1 << 0)) ? o_w_gid_off : lid % c_w;
                int o_h = (bitmap & (1 << 1)) ? o_h_gid_off : (lid / o_h_div) % c_h;
                int o_c = (bitmap & (1 << 2)) ? o_c_gid_off : (lid / o_c_div) % c_c;
                int o_n = (bitmap & (1 << 3)) ? o_n_gid_off : lid / o_n_div;

                int aindex    = o_n * a_nstride + o_c * a_cstride + o_h * a_hstride + o_w;
                int cindex    = o_n * c_nstride + o_c * c_cstride + o_h * c_hstride + o_w;
                c_off[cindex] = MIOPEN_TENSOR_OP(a_off[aindex] * alpha0, operand);

                lid += get_local_size(0);
            }
        }
    }
    else
    {
        for(; gid < num_wg; gid += MAX_NUM_WG)
        {
            int lid = get_local_id(0);

            int o_h_div = bitmap & (1 << 0) ? 1 : c_w;
            int o_c_div = o_h_div * (bitmap & (1 << 1) ? 1 : c_h);
            int o_n_div = o_c_div * (bitmap & (1 << 2) ? 1 : c_c);

            int o_w_gid_off = gid % b_w;
            int o_h_gid_off = (gid / b_w) % b_h;
            int o_c_gid_off = (gid / b_w / b_h) % b_c;
            int o_n_gid_off = (gid / b_w / b_h) / b_c;

            int bindex = o_n_gid_off * b_nstride + o_c_gid_off * b_cstride +
                         o_h_gid_off * b_hstride + o_w_gid_off;
            MIOPEN_TYPE operand = b_off[bindex] * alpha1;

            while(lid < work_per_wg)
            {
                int o_w = (bitmap & (1 << 0)) ? o_w_gid_off : lid % c_w;
                int o_h = (bitmap & (1 << 1)) ? o_h_gid_off : (lid / o_h_div) % c_h;
                int o_c = (bitmap & (1 << 2)) ? o_c_gid_off : (lid / o_c_div) % c_c;
                int o_n = (bitmap & (1 << 3)) ? o_n_gid_off : lid / o_n_div;

                int aindex = o_n * a_nstride + o_c * a_cstride + o_h * a_hstride + o_w;
                int cindex = o_n * c_nstride + o_c * c_cstride + o_h * c_hstride + o_w;
                c_off[cindex] =
                    MIOPEN_TENSOR_OP(a_off[aindex] * alpha0, operand) + beta * c_off[cindex];

                lid += get_local_size(0);
            }
        }
    }
}

#endif

#ifdef USE_5D_TENSOR_GENERIC
// NCDHW
// (samples, color_depth, frames, width, height )
__kernel void Op5dTensorGeneric(global MIOPEN_TYPE* a,
                                const int a_nstride,
                                const int a_cstride,
                                const int a_dstride,
                                const int a_hstride,
                                global MIOPEN_TYPE* b,
                                const int b_c,
                                const int b_d,
                                const int b_h,
                                const int b_w,
                                const int b_nstride,
                                const int b_cstride,
                                const int b_dstride,
                                const int b_hstride,
                                global MIOPEN_TYPE* c,
                                const int c_c,
                                const int c_d,
                                const int c_h,
                                const int c_w,
                                const int c_nstride,
                                const int c_cstride,
                                const int c_dstride,
                                const int c_hstride,
                                const MIOPEN_TYPE alpha0,
                                const MIOPEN_TYPE alpha1,
                                const MIOPEN_TYPE beta,
                                const unsigned int bitmap,
                                const int work_per_wg,
                                const long Aoffset,
                                const long Boffset,
                                const long Coffset,
                                const int num_wg)
{
    int gid = get_group_id(0);

    global MIOPEN_TYPE* a_off = a + Aoffset;
    global MIOPEN_TYPE* b_off = b + Boffset;
    global MIOPEN_TYPE* c_off = c + Coffset;

    // num_wg: the number of workgroups should be launched
    // MAX_NUM_WG: the maximum number of workgroups actually launched
    if(beta == (MIOPEN_TYPE)0)
    {
        for(; gid < num_wg; gid += MAX_NUM_WG)
        {

            int lid = get_local_id(0);

            int o_h_div = bitmap & (1 << 0) ? 1 : c_w;
            int o_d_div = o_h_div * (bitmap & (1 << 1) ? 1 : c_h);
            int o_c_div = o_d_div * (bitmap & (1 << 2) ? 1 : c_d);
            int o_n_div = o_c_div * (bitmap & (1 << 3) ? 1 : c_c);

            int o_w_gid_off = gid % b_w;
            int o_h_gid_off = (gid / b_w) % b_h;
            int o_d_gid_off = (gid / b_w / b_h) % b_d;
            int o_c_gid_off = (gid / b_w / b_h / b_d) % b_c;
            int o_n_gid_off = (gid / b_w / b_h / b_d) / b_c;

            int bindex = o_n_gid_off * b_nstride + o_c_gid_off * b_cstride +
                         o_d_gid_off * b_dstride + o_h_gid_off * b_hstride + o_w_gid_off;

            MIOPEN_TYPE operand = b_off[bindex] * alpha1;

            while(lid < work_per_wg)
            {
                int o_w = (bitmap & (1 << 0)) ? o_w_gid_off : lid % c_w;
                int o_h = (bitmap & (1 << 1)) ? o_h_gid_off : (lid / o_h_div) % c_h;
                int o_d = (bitmap & (1 << 2)) ? o_d_gid_off : (lid / o_d_div) % c_d;
                int o_c = (bitmap & (1 << 3)) ? o_c_gid_off : (lid / o_c_div) % c_c;
                int o_n = (bitmap & (1 << 4)) ? o_n_gid_off : lid / o_n_div;

                int aindex =
                    o_n * a_nstride + o_c * a_cstride + o_d * a_dstride + o_h * a_hstride + o_w;
                int cindex =
                    o_n * c_nstride + o_c * c_cstride + o_d * c_dstride + o_h * c_hstride + o_w;

                c_off[cindex] = MIOPEN_TENSOR_OP(a_off[aindex] * alpha0, operand);

                lid += get_local_size(0);
            }
        }
    }
    else
    {
        for(; gid < num_wg; gid += MAX_NUM_WG)
        {

            int lid = get_local_id(0);

            int o_h_div = bitmap & (1 << 0) ? 1 : c_w;
            int o_d_div = o_h_div * (bitmap & (1 << 1) ? 1 : c_h);
            int o_c_div = o_d_div * (bitmap & (1 << 2) ? 1 : c_d);
            int o_n_div = o_c_div * (bitmap & (1 << 3) ? 1 : c_c);

            int o_w_gid_off = gid % b_w;
            int o_h_gid_off = (gid / b_w) % b_h;
            int o_d_gid_off = (gid / b_w / b_h) % b_d;
            int o_c_gid_off = (gid / b_w / b_h / b_d) % b_c;
            int o_n_gid_off = (gid / b_w / b_h / b_d) / b_c;

            int bindex = o_n_gid_off * b_nstride + o_c_gid_off * b_cstride +
                         o_d_gid_off * b_dstride + o_h_gid_off * b_hstride + o_w_gid_off;

            MIOPEN_TYPE operand = b_off[bindex] * alpha1;

            while(lid < work_per_wg)
            {
                int o_w = (bitmap & (1 << 0)) ? o_w_gid_off : lid % c_w;
                int o_h = (bitmap & (1 << 1)) ? o_h_gid_off : (lid / o_h_div) % c_h;
                int o_d = (bitmap & (1 << 2)) ? o_d_gid_off : (lid / o_d_div) % c_d;
                int o_c = (bitmap & (1 << 3)) ? o_c_gid_off : (lid / o_c_div) % c_c;
                int o_n = (bitmap & (1 << 4)) ? o_n_gid_off : lid / o_n_div;

                int aindex =
                    o_n * a_nstride + o_c * a_cstride + o_d * a_dstride + o_h * a_hstride + o_w;
                int cindex =
                    o_n * c_nstride + o_c * c_cstride + o_d * c_dstride + o_h * c_hstride + o_w;

                c_off[cindex] =
                    MIOPEN_TENSOR_OP(a_off[aindex] * alpha0, operand) + beta * c_off[cindex];

                lid += get_local_size(0);
            }
        }
    }
}

#endif

#ifdef USE_2D_TENSOR_LITE
__kernel void Op2dTensorLite(const global MIOPEN_TYPE* a,
                             const int a_nstride,
                             const global MIOPEN_TYPE* b,
                             const int b_nstride,
                             global MIOPEN_TYPE* c,
                             const int c_nstride,
                             const MIOPEN_TYPE alpha0,
                             const MIOPEN_TYPE alpha1,
                             const MIOPEN_TYPE beta,
                             const long Aoffset,
                             const long Boffset,
                             const long Coffset,
                             const long total_work,
                             const long total_work2,
                             const int use_beta,
                             const int use_bias)
{
    int gid0 = get_global_id(0);
    int gid1 = get_global_id(1);

    MIOPEN_TYPE a_dat[RD_BLCK];
    MIOPEN_TYPE b_dat[RD_BLCK];
    MIOPEN_TYPE c_dat[RD_BLCK];

    if(gid0 < total_work)
    {
        if(use_bias == 1)
        {
            int b_index          = gid0 * RD_BLCK;
            *((READ_TYPE*)b_dat) = *((const global READ_TYPE*)(b + Boffset + b_index));
        }

        if(beta == (MIOPEN_TYPE)0)
        {
            for(; gid1 < total_work2; gid1 += get_global_size(1))
            {
                for(int i = 0; i < RD_BLCK; ++i)
                {
                    c_dat[i] = (MIOPEN_TYPE)0;
                }

                int a_index = gid1 * a_nstride + gid0 * RD_BLCK;
                int c_index = gid1 * c_nstride + gid0 * RD_BLCK;

                *((READ_TYPE*)a_dat) = *((const global READ_TYPE*)(a + Aoffset + a_index));
                if(use_beta == 1)
                {
                    *((READ_TYPE*)c_dat) = *((const global READ_TYPE*)(c + Coffset + c_index));
                }

                if(use_bias == 0)
                {
                    int b_index          = gid1 * b_nstride + gid0 * RD_BLCK;
                    *((READ_TYPE*)b_dat) = *((const global READ_TYPE*)(b + Boffset + b_index));
                }

                for(int i = 0; i < RD_BLCK; ++i)
                {
                    if(use_beta == 1)
                    {
                        c_dat[i] = (MIOPEN_TYPE)0;
                    }
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i] * alpha0, b_dat[i] * alpha1);
                }

                *((global READ_TYPE*)(c + Coffset + c_index)) = *((READ_TYPE*)c_dat);
            }
        }
        else
        {
            for(; gid1 < total_work2; gid1 += get_global_size(1))
            {
                for(int i = 0; i < RD_BLCK; ++i)
                {
                    c_dat[i] = (MIOPEN_TYPE)0;
                }

                int a_index = gid1 * a_nstride + gid0 * RD_BLCK;
                int c_index = gid1 * c_nstride + gid0 * RD_BLCK;

                *((READ_TYPE*)a_dat) = *((const global READ_TYPE*)(a + Aoffset + a_index));
                if(use_beta == 1)
                {
                    *((READ_TYPE*)c_dat) = *((const global READ_TYPE*)(c + Coffset + c_index));
                }

                if(use_bias == 0)
                {
                    int b_index          = gid1 * b_nstride + gid0 * RD_BLCK;
                    *((READ_TYPE*)b_dat) = *((const global READ_TYPE*)(b + Boffset + b_index));
                }

                for(int i = 0; i < RD_BLCK; ++i)
                {
                    if(use_beta == 1)
                    {
                        c_dat[i] *= beta;
                    }
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i] * alpha0, b_dat[i] * alpha1);
                }

                *((global READ_TYPE*)(c + Coffset + c_index)) = *((READ_TYPE*)c_dat);
            }
        }
    }
}

#endif

#ifdef USE_2D_TENSOR_SQUASH
__kernel void Op2dTensorSquash(const global MIOPEN_TYPE* a,
                               const global MIOPEN_TYPE* b,
                               const int b_c,
                               const int b_nstride,
                               global MIOPEN_TYPE* c,
                               const MIOPEN_TYPE alpha0,
                               const MIOPEN_TYPE alpha1,
                               const MIOPEN_TYPE beta,
                               const long Aoffset,
                               const long Boffset,
                               const long Coffset,
                               const long total_work,
                               const int use_apl0,
                               const int use_apl1,
                               const int use_bet)
{
    MIOPEN_TYPE a_dat[RD_BLCK];
    MIOPEN_TYPE b_dat1[RD_BLCK];
    MIOPEN_TYPE b_dat2[RD_BLCK];
    MIOPEN_TYPE b_dat3[RD_BLCK];
    MIOPEN_TYPE b_dat4[RD_BLCK];
    MIOPEN_TYPE b_dat5[RD_BLCK];
    MIOPEN_TYPE b_dat6[RD_BLCK];
    MIOPEN_TYPE b_dat7[RD_BLCK];
    MIOPEN_TYPE b_dat8[RD_BLCK];
    MIOPEN_TYPE b_dat9[RD_BLCK];
    MIOPEN_TYPE b_dat10[RD_BLCK];
    MIOPEN_TYPE b_dat11[RD_BLCK];
    MIOPEN_TYPE b_dat12[RD_BLCK];
    MIOPEN_TYPE b_dat13[RD_BLCK];
    MIOPEN_TYPE b_dat14[RD_BLCK];
    MIOPEN_TYPE b_dat15[RD_BLCK];
    MIOPEN_TYPE b_dat16[RD_BLCK];
    MIOPEN_TYPE c_dat[RD_BLCK];
    int g_RD_BLCK;

    for(int i = 0; i < RD_BLCK; ++i)
    {
        b_dat1[i]  = (MIOPEN_TYPE)0;
        b_dat2[i]  = (MIOPEN_TYPE)0;
        b_dat3[i]  = (MIOPEN_TYPE)0;
        b_dat4[i]  = (MIOPEN_TYPE)0;
        b_dat5[i]  = (MIOPEN_TYPE)0;
        b_dat6[i]  = (MIOPEN_TYPE)0;
        b_dat7[i]  = (MIOPEN_TYPE)0;
        b_dat8[i]  = (MIOPEN_TYPE)0;
        b_dat9[i]  = (MIOPEN_TYPE)0;
        b_dat10[i] = (MIOPEN_TYPE)0;
        b_dat11[i] = (MIOPEN_TYPE)0;
        b_dat12[i] = (MIOPEN_TYPE)0;
        b_dat13[i] = (MIOPEN_TYPE)0;
        b_dat14[i] = (MIOPEN_TYPE)0;
        b_dat15[i] = (MIOPEN_TYPE)0;
        b_dat16[i] = (MIOPEN_TYPE)0;
    }

    for(int gid = get_global_id(0); gid < total_work; gid += get_global_size(0))
    {
        for(int i = 0; i < RD_BLCK; ++i)
        {
            a_dat[i] = (MIOPEN_TYPE)0;
            c_dat[i] = (MIOPEN_TYPE)0;
        }

        int io_index = gid * RD_BLCK;
        if(use_apl0 == 1)
        {
            *((READ_TYPE*)a_dat) = *((const global READ_TYPE*)(a + Aoffset + io_index));
            for(int i = 0; i < RD_BLCK; ++i)
            {
                a_dat[i] *= alpha0;
            }
        }

        if(use_bet == 1)
        {
            *((READ_TYPE*)c_dat) = *((const global READ_TYPE*)(c + Coffset + io_index));
            if(beta == (MIOPEN_TYPE)0)
            {
                for(int i = 0; i < RD_BLCK; ++i)
                {
                    c_dat[i] = 0;
                }
            }
            else
            {
                for(int i = 0; i < RD_BLCK; ++i)
                {
                    c_dat[i] *= beta;
                }
            }
        }

        g_RD_BLCK = gid * RD_BLCK;
        if(use_apl1 == 1)
        {
            for(int bid = 0; bid < ((b_c / 16) * 16); bid += 16)
            {

                int b_index1           = (bid * b_nstride) + g_RD_BLCK;
                int b_index2           = ((bid + 1) * b_nstride) + g_RD_BLCK;
                int b_index3           = ((bid + 2) * b_nstride) + g_RD_BLCK;
                int b_index4           = ((bid + 3) * b_nstride) + g_RD_BLCK;
                int b_index5           = ((bid + 4) * b_nstride) + g_RD_BLCK;
                int b_index6           = ((bid + 5) * b_nstride) + g_RD_BLCK;
                int b_index7           = ((bid + 6) * b_nstride) + g_RD_BLCK;
                int b_index8           = ((bid + 7) * b_nstride) + g_RD_BLCK;
                int b_index9           = ((bid + 8) * b_nstride) + g_RD_BLCK;
                int b_index10          = ((bid + 9) * b_nstride) + g_RD_BLCK;
                int b_index11          = ((bid + 10) * b_nstride) + g_RD_BLCK;
                int b_index12          = ((bid + 11) * b_nstride) + g_RD_BLCK;
                int b_index13          = ((bid + 12) * b_nstride) + g_RD_BLCK;
                int b_index14          = ((bid + 13) * b_nstride) + g_RD_BLCK;
                int b_index15          = ((bid + 14) * b_nstride) + g_RD_BLCK;
                int b_index16          = ((bid + 15) * b_nstride) + g_RD_BLCK;
                *((READ_TYPE*)b_dat1)  = *((const global READ_TYPE*)(b + Boffset + b_index1));
                *((READ_TYPE*)b_dat2)  = *((const global READ_TYPE*)(b + Boffset + b_index2));
                *((READ_TYPE*)b_dat3)  = *((const global READ_TYPE*)(b + Boffset + b_index3));
                *((READ_TYPE*)b_dat4)  = *((const global READ_TYPE*)(b + Boffset + b_index4));
                *((READ_TYPE*)b_dat5)  = *((const global READ_TYPE*)(b + Boffset + b_index5));
                *((READ_TYPE*)b_dat6)  = *((const global READ_TYPE*)(b + Boffset + b_index6));
                *((READ_TYPE*)b_dat7)  = *((const global READ_TYPE*)(b + Boffset + b_index7));
                *((READ_TYPE*)b_dat8)  = *((const global READ_TYPE*)(b + Boffset + b_index8));
                *((READ_TYPE*)b_dat9)  = *((const global READ_TYPE*)(b + Boffset + b_index9));
                *((READ_TYPE*)b_dat10) = *((const global READ_TYPE*)(b + Boffset + b_index10));
                *((READ_TYPE*)b_dat11) = *((const global READ_TYPE*)(b + Boffset + b_index11));
                *((READ_TYPE*)b_dat12) = *((const global READ_TYPE*)(b + Boffset + b_index12));
                *((READ_TYPE*)b_dat13) = *((const global READ_TYPE*)(b + Boffset + b_index13));
                *((READ_TYPE*)b_dat14) = *((const global READ_TYPE*)(b + Boffset + b_index14));
                *((READ_TYPE*)b_dat15) = *((const global READ_TYPE*)(b + Boffset + b_index15));
                *((READ_TYPE*)b_dat16) = *((const global READ_TYPE*)(b + Boffset + b_index16));

                for(int i = 0; i < RD_BLCK; ++i)
                {
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat1[i] * alpha1);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat2[i] * alpha1);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat3[i] * alpha1);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat4[i] * alpha1);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat5[i] * alpha1);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat6[i] * alpha1);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat7[i] * alpha1);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat8[i] * alpha1);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat9[i] * alpha1);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat10[i] * alpha1);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat11[i] * alpha1);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat12[i] * alpha1);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat13[i] * alpha1);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat14[i] * alpha1);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat15[i] * alpha1);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat16[i] * alpha1);
                }
            }
            for(int bid = ((b_c / 16) * 16); bid < b_c; bid++)
            {
                int b_index           = bid * b_nstride + g_RD_BLCK;
                *((READ_TYPE*)b_dat1) = *((const global READ_TYPE*)(b + Boffset + b_index));

                for(int i = 0; i < RD_BLCK; ++i)
                {
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat1[i] * alpha1);
                }
            }
        }
        else
        {
            for(int bid = 0; bid < ((b_c / 16) * 16); bid += 16)
            {

                for(int i = 0; i < RD_BLCK; ++i)
                {
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], 0);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], 0);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], 0);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], 0);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], 0);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], 0);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], 0);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], 0);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], 0);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], 0);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], 0);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], 0);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], 0);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], 0);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], 0);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], 0);
                }
            }
            for(int bid = ((b_c / 16) * 16); bid < b_c; bid++)
            {

                for(int i = 0; i < RD_BLCK; ++i)
                {
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], 0);
                }
            }
        }
        *((global READ_TYPE*)(c + Coffset + io_index)) = *((READ_TYPE*)c_dat);
    }
}
#endif

#ifdef USE_4D_TENSOR_LITE
// N - batch size
// C - # of maps
// H - map height
// W - map width
// TENS_LEN = (N*C*H*W);
// RD_BLCK = (TENS_LEN%4==0) ? 4 : (TENS_LEN%3==0)? 3 : (TENS_LEN%2==0)? 2 : 1;
// READ_TYPE = (RD_BLCK==4) ? "float4" : (RD_BLCK == 3) ? "float3" : (RD_BLC==2) ? "float2" :
// "float";
// local size = (256, 1, 1)
// global size = ((TENS_LEN/RD_BLCK), 1, 1)
__kernel void Op4dTensorLite(const global MIOPEN_TYPE* a,
                             const global MIOPEN_TYPE* b,
                             global MIOPEN_TYPE* c,
                             const MIOPEN_TYPE alpha0,
                             const MIOPEN_TYPE alpha1,
                             const MIOPEN_TYPE beta,
                             const long Aoffset,
                             const long Boffset,
                             const long Coffset,
                             const long total_work,
                             const int use_beta)
{
    int gid0 = get_global_id(0);

    MIOPEN_TYPE a_dat[RD_BLCK];
    MIOPEN_TYPE b_dat[RD_BLCK];
    MIOPEN_TYPE c_dat[RD_BLCK];

    if(beta == (MIOPEN_TYPE)0)
    {
        for(; gid0 < total_work; gid0 += get_global_size(0))
        {
            int index = gid0 * RD_BLCK;

            for(int i = 0; i < RD_BLCK; ++i)
            {
                c_dat[i] = (MIOPEN_TYPE)0;
            }

            *((READ_TYPE*)a_dat) = *((const global READ_TYPE*)(a + index + Aoffset));
            *((READ_TYPE*)b_dat) = *((const global READ_TYPE*)(b + index + Boffset));
            if(use_beta == 1)
            {
                *((READ_TYPE*)c_dat) = *((const global READ_TYPE*)(c + index + Coffset));
            }

            for(int i = 0; i < RD_BLCK; ++i)
            {
                if(use_beta == 1)
                {
                    c_dat[i] = (MIOPEN_TYPE)0;
                }
                c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i] * alpha0, b_dat[i] * alpha1);
            }

            *((global READ_TYPE*)(c + index + Coffset)) = *((READ_TYPE*)c_dat);
        }
    }
    else
    {
        for(; gid0 < total_work; gid0 += get_global_size(0))
        {
            int index = gid0 * RD_BLCK;

            for(int i = 0; i < RD_BLCK; ++i)
            {
                c_dat[i] = (MIOPEN_TYPE)0;
            }

            *((READ_TYPE*)a_dat) = *((const global READ_TYPE*)(a + index + Aoffset));
            *((READ_TYPE*)b_dat) = *((const global READ_TYPE*)(b + index + Boffset));
            if(use_beta == 1)
            {
                *((READ_TYPE*)c_dat) = *((const global READ_TYPE*)(c + index + Coffset));
            }

            for(int i = 0; i < RD_BLCK; ++i)
            {
                if(use_beta == 1)
                {
                    c_dat[i] *= beta;
                }
                c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i] * alpha0, b_dat[i] * alpha1);
            }

            *((global READ_TYPE*)(c + index + Coffset)) = *((READ_TYPE*)c_dat);
        }
    }
}

#endif
