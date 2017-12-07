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

#ifndef MIOPEN_TENSOR_DIMS
#define MIOPEN_TENSOR_DIMS 4
#endif

#define UNUSED __attribute__((__unused__))

MIOPEN_TYPE miopenAdd(MIOPEN_TYPE a, MIOPEN_TYPE b) { return a + b; }

MIOPEN_TYPE miopenMul(MIOPEN_TYPE a, MIOPEN_TYPE b) { return a * b; }

MIOPEN_TYPE miopenMax(MIOPEN_TYPE a, MIOPEN_TYPE b) { return ((a > b) ? a : b); }

MIOPEN_TYPE miopenMin(MIOPEN_TYPE a, MIOPEN_TYPE b) { return ((a < b) ? a : b); }

__kernel void OpTensorFwdBias(global MIOPEN_TYPE* a,
                              global MIOPEN_TYPE* b,
#if INCR_WG == 0
                              UNUSED
#endif
                              const int b_c,
                              global MIOPEN_TYPE* c,
#if INCR_WG == 1
                              UNUSED
#endif
                              const int c_n,
                              const int c_nstride,
                              const int c_cstride,
                              const int work_per_wg,
                              const float alpha0,
                              const float alpha1,
                              const float beta,
                              const long Aoffset,
                              const long Boffset,
                              const long Coffset,
                              const int num_wg)
{
    global MIOPEN_TYPE* a_off = a + Aoffset;
    global MIOPEN_TYPE* b_off = b + Boffset;
    global MIOPEN_TYPE* c_off = c + Coffset;

    int gid = get_group_id(0);
    int lid = get_local_id(0);

    // num_wg: the number of workgroups should be launched
    // MAX_NUM_WG: the maximum number of workgroups actually launched
    for(; gid < num_wg; gid += MAX_NUM_WG)
    {

#if INCR_WG == 1
        int o_n             = gid / b_c;
        int o_c             = gid % b_c;
        MIOPEN_TYPE operand = b_off[o_c] * alpha1;

        while(lid < work_per_wg)
        {
            int index    = o_n * c_nstride + o_c * c_cstride + lid;
            c_off[index] = MIOPEN_TENSOR_OP(a_off[index] * alpha0, operand) + beta * c_off[index];
            lid += get_local_size(0);
        }

// each workgroup computes N*H*W for each C (bias-term)
// number of workgroups = c_c (b_c)
#elif INCR_WG == 0
        MIOPEN_TYPE operand = b_off[gid] * alpha1;
        int work_off        = work_per_wg / c_n;

        while(lid < work_per_wg)
        {
            int o_hw     = lid % work_off;
            int o_n      = lid / work_off;
            int index    = o_n * c_nstride + gid * c_cstride + o_hw;
            c_off[index] = MIOPEN_TENSOR_OP(a_off[index] * alpha0, operand) + beta * c_off[index];

            lid += get_local_size(0);
        }
#endif // INCR_WG
    }
}

__kernel void OpTensorFwdBiasGeneric(global MIOPEN_TYPE* a,
                                     const int a_nstride,
                                     const int a_cstride,
                                     const int a_hstride,
                                     global MIOPEN_TYPE* b,
#if INCR_WG == 0
                                     UNUSED
#endif
                                     const int b_c,
                                     const int b_cstride,
                                     global MIOPEN_TYPE* c,
#if INCR_WG == 1
                                     UNUSED
#endif
                                     const int c_n,
                                     const int c_w,
                                     const int c_nstride,
                                     const int c_cstride,
                                     const int c_hstride,
                                     const float alpha0,
                                     const float alpha1,
                                     const float beta,
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
    for(; gid < num_wg; gid += MAX_NUM_WG)
    {
        int lid = get_local_id(0);

#if INCR_WG == 1

        int o_c = gid % b_c;
        int o_n = gid / b_c;

        int bindex          = o_c * b_cstride;
        MIOPEN_TYPE operand = b_off[bindex] * alpha1;

        while(lid < work_per_wg)
        {
            int o_h    = lid / c_w;
            int o_w    = lid % c_w;
            int aindex = o_n * a_nstride + o_c * a_cstride + o_h * a_hstride + o_w;
            int cindex = o_n * c_nstride + o_c * c_cstride + o_h * c_hstride + o_w;
            c_off[cindex] =
                MIOPEN_TENSOR_OP(a_off[aindex] * alpha0, operand) + beta * c_off[cindex];
            lid += get_local_size(0);
        }

// each workgroup computes N*H*W for each C (bias-term)
// number of workgroups = c_c (b_c)
#elif INCR_WG == 0
        MIOPEN_TYPE operand = b_off[gid * b_cstride] * alpha1;
        // int work_off        = work_per_wg / c_n;

        while(lid < work_per_wg)
        {
            int o_n    = lid % c_n;
            int o_h    = (lid / c_n) / c_w;
            int o_w    = (lid / c_n) % c_w;
            int aindex = o_n * a_nstride + gid * a_cstride + o_h * a_hstride + o_w;
            int cindex = o_n * c_nstride + gid * c_cstride + o_h * c_hstride + o_w;
            c_off[cindex] =
                MIOPEN_TENSOR_OP(a_off[aindex] * alpha0, operand) + beta * c_off[cindex];

            lid += get_local_size(0);
        }
#endif // INCR_WG
    }
}

// DLOWELL : cutting out this section
#if(FIRST_NOT_ONE < 4 && MIOPEN_TENSOR_DIMS == 4)

__kernel void OpTensorLeadingOnes(global MIOPEN_TYPE* a,
                                  global MIOPEN_TYPE* b,
                                  global MIOPEN_TYPE* c,
#if FIRST_NOT_ONE == 0
                                  UNUSED
#endif
                                  const int c_c,
#if FIRST_NOT_ONE <= 1
                                  UNUSED
#endif
                                  const int c_h,
#if FIRST_NOT_ONE <= 1
                                  UNUSED
#endif
                                  const int c_w,
                                  const int c_nstride,
#if FIRST_NOT_ONE == 0
                                  UNUSED
#endif
                                  const int c_cstride,
#if FIRST_NOT_ONE == 3
                                  UNUSED
#endif
                                  const int work_per_wg,
                                  const float alpha0,
                                  const float alpha1,
                                  const float beta,
                                  const long Aoffset,
                                  const long Boffset,
                                  const long Coffset,
                                  const int num_wg)
{

    /* Special case for leading ones where the total no. of threads is the
     * inner_product of the tensor dims.  Each thread just updates one value
     */

    global MIOPEN_TYPE* a_off = a + Aoffset;
    global MIOPEN_TYPE* b_off = b + Boffset;
    global MIOPEN_TYPE* c_off = c + Coffset;

#if FIRST_NOT_ONE == 3 // bitmap = 1,1,1,1

    int tid = get_global_id(0);
    // num_wg: the number of workgroups should be launched
    // MAX_NUM_WG: the maximum number of workgroups actually launched
    for(; tid < num_wg; tid += MAX_NUM_WG)
    {

        MIOPEN_TYPE operand = b_off[tid] * alpha1;

        int o_w = tid % c_w;
        int o_h = (tid / c_w) % c_h;
        int o_c = (tid / (c_w * c_h)) % c_c;
        int o_n = tid / (c_w * c_h * c_c);

        int index    = o_n * c_nstride + o_c * c_cstride + o_h * c_w + o_w;
        c_off[index] = MIOPEN_TENSOR_OP(a_off[index] * alpha0, operand) + beta * c_off[index];
    }
#elif FIRST_NOT_ONE == 2 // bitmap = 1,1,1,0
    int gid = get_group_id(0);
    // num_wg: the number of workgroups should be launched
    // MAX_NUM_WG: the maximum number of workgroups actually launched
    for(; gid < num_wg; gid += MAX_NUM_WG)
    {

        int lid             = get_local_id(0);
        MIOPEN_TYPE operand = b_off[gid] * alpha1;

        int o_h = gid % c_h;
        int o_c = (gid / c_h) % c_c;
        int o_n = gid / (c_c * c_h);

        while(lid < work_per_wg)
        {
            int index    = o_n * c_nstride + o_c * c_cstride + o_h * c_w + lid;
            c_off[index] = MIOPEN_TENSOR_OP(a_off[index] * alpha0, operand) + beta * c_off[index];
            lid += get_local_size(0);
        }
    }
#elif FIRST_NOT_ONE == 1 // bitmap = 1,1,0,0
    int gid = get_group_id(0);
    // num_wg: the number of workgroups should be launched
    // MAX_NUM_WG: the maximum number of workgroups actually launched
    for(; gid < num_wg; gid += MAX_NUM_WG)
    {

        int lid             = get_local_id(0);
        MIOPEN_TYPE operand = b_off[gid] * alpha1;

        int o_c = gid % c_c;
        int o_n = gid / c_c;

        while(lid < work_per_wg)
        {
            int index    = o_n * c_nstride + o_c * c_cstride + lid;
            c_off[index] = MIOPEN_TENSOR_OP(a_off[index] * alpha0, operand) + beta * c_off[index];
            lid += get_local_size(0);
        }
    }

#elif FIRST_NOT_ONE == 0 // bitmap = 1,0,0,0
    int gid = get_group_id(0);
    // num_wg: the number of workgroups should be launched
    // MAX_NUM_WG: the maximum number of workgroups actually launched
    for(; gid < num_wg; gid += MAX_NUM_WG)
    {
        int lid             = get_local_id(0);
        MIOPEN_TYPE operand = b_off[gid] * alpha1;

        while(lid < work_per_wg)
        {
            int index    = gid * c_nstride + lid;
            c_off[index] = MIOPEN_TENSOR_OP(a_off[index] * alpha0, operand) + beta * c_off[index];
            lid += get_local_size(0);
        }
    }
#endif
}

#endif

// DLOWELL : cutting out this section
#if(FIRST_NOT_ONE < 4 && MIOPEN_TENSOR_DIMS == 4)

__kernel void OpTensorLeadingOnesGeneric(global MIOPEN_TYPE* a,
                                         const int a_nstride,
                                         const int a_cstride,
                                         const int a_hstride,
                                         global MIOPEN_TYPE* b,
                                         const int b_nstride,
#if FIRST_NOT_ONE == 0
                                         UNUSED
#endif
                                         const int b_cstride,
#if FIRST_NOT_ONE <= 1
                                         UNUSED
#endif
                                         const int b_hstride,
                                         global MIOPEN_TYPE* c,
                                         const int c_c,
#if FIRST_NOT_ONE == 1
                                         UNUSED
#endif
                                         const int c_h,
#if FIRST_NOT_ONE == 0 || FIRST_NOT_ONE == 2
                                         UNUSED
#endif
                                         const int c_w,
                                         const int c_nstride,
                                         const int c_cstride,
                                         const int c_hstride,
                                         const float alpha0,
                                         const float alpha1,
                                         const float beta,
#if FIRST_NOT_ONE == 3
                                         UNUSED
#endif
                                         const int work_per_wg,
                                         const long Aoffset,
                                         const long Boffset,
                                         const long Coffset,
                                         const int num_wg)
{

    /* Special case for leading ones where the total no. of threads is the
     * inner_product of the tensor dims.  Each thread just updates one value
     */
    global MIOPEN_TYPE* a_off = a + Aoffset;
    global MIOPEN_TYPE* b_off = b + Boffset;
    global MIOPEN_TYPE* c_off = c + Coffset;
#if FIRST_NOT_ONE == 3 // bitmap = 1,1,1,1
    int tid = get_global_id(0);
    // MIOPEN_TYPE operand = b[tid];

    // num_wg: the number of workgroups should be launched
    // MAX_NUM_WG: the maximum number of workgroups actually launched
    for(; tid < num_wg; tid += MAX_NUM_WG)
    {

        int o_w = tid % c_w;
        int o_h = (tid / c_w) % c_h;
        int o_c = (tid / (c_w * c_h)) % c_c;
        int o_n = tid / (c_w * c_h * c_c);

        int aindex = o_n * a_nstride + o_c * a_cstride + o_h * a_hstride + o_w;
        int bindex = o_n * b_nstride + o_c * b_cstride + o_h * b_hstride + o_w;
        int cindex = o_n * c_nstride + o_c * c_cstride + o_h * c_hstride + o_w;
        c_off[cindex] =
            MIOPEN_TENSOR_OP(alpha0 * a_off[aindex], alpha1 * b_off[bindex]) + beta * c_off[cindex];
    }

#elif FIRST_NOT_ONE == 2 // bitmap = 1,1,1,0
    int gid = get_group_id(0);

    // num_wg: the number of workgroups should be launched
    // MAX_NUM_WG: the maximum number of workgroups actually launched
    for(; gid < num_wg; gid += MAX_NUM_WG)
    {
        int lid = get_local_id(0);
        int o_h = gid % c_h;
        int o_c = (gid / c_h) % c_c;
        int o_n = gid / (c_c * c_h);

        int bindex          = o_n * b_nstride + o_c * b_cstride + o_h * b_hstride;
        MIOPEN_TYPE operand = b_off[bindex] * alpha1;

        while(lid < work_per_wg)
        {
            int aindex = o_n * a_nstride + o_c * a_cstride + o_h * a_hstride + lid;
            int cindex = o_n * c_nstride + o_c * c_cstride + o_h * c_hstride + lid;
            c_off[cindex] =
                MIOPEN_TENSOR_OP(alpha0 * a_off[aindex], operand) + beta * c_off[cindex];

            lid += get_local_size(0);
        }
    }
#elif FIRST_NOT_ONE == 1 // bitmap = 1,1,0,0
    int gid = get_group_id(0);

    // num_wg: the number of workgroups should be launched
    // MAX_NUM_WG: the maximum number of workgroups actually launched
    for(; gid < num_wg; gid += MAX_NUM_WG)
    {
        int lid = get_local_id(0);
        int o_c = gid % c_c;
        int o_n = gid / c_c;

        int bindex          = o_n * b_nstride + o_c * b_cstride;
        MIOPEN_TYPE operand = b_off[bindex] * alpha1;

        while(lid < work_per_wg)
        {
            int o_h    = lid / c_w;
            int o_w    = lid % c_w;
            int aindex = o_n * a_nstride + o_c * a_cstride + o_h * a_hstride + o_w;
            int cindex = o_n * c_nstride + o_c * c_cstride + o_h * c_hstride + o_w;
            c_off[cindex] =
                MIOPEN_TENSOR_OP(a_off[aindex] * alpha0, operand) + beta * c_off[cindex];
            lid += get_local_size(0);
        }
    }

#elif FIRST_NOT_ONE == 0 // bitmap = 1,0,0,0
    int gid = get_group_id(0);

    // num_wg: the number of workgroups should be launched
    // MAX_NUM_WG: the maximum number of workgroups actually launched
    for(; gid < num_wg; gid += MAX_NUM_WG)
    {
        int lid             = get_local_id(0);
        MIOPEN_TYPE operand = b_off[gid * b_nstride] * alpha1;

        while(lid < work_per_wg)
        {
            int o_c    = lid % c_c;
            int o_h    = (lid / c_c) % c_h;
            int o_w    = (lid / c_c) / c_h;
            int aindex = gid * a_nstride + o_c * a_cstride + o_h * a_hstride + o_w;
            int cindex = gid * c_nstride + o_c * c_cstride + o_h * c_hstride + o_w;
            c_off[cindex] =
                MIOPEN_TENSOR_OP(a_off[aindex] * alpha0, operand) + beta * c_off[cindex];

            lid += get_local_size(0);
        }
    }
#endif
}

#endif

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
                                const float alpha0,
                                const float alpha1,
                                const float beta,
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

        int bindex = o_n_gid_off * b_nstride + o_c_gid_off * b_cstride + o_h_gid_off * b_hstride +
                     o_w_gid_off;
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
                                const float alpha0,
                                const float alpha1,
                                const float beta,
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

        int bindex = o_n_gid_off * b_nstride + o_c_gid_off * b_cstride + o_d_gid_off * b_dstride +
                     o_h_gid_off * b_hstride + o_w_gid_off;

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

// NCH
__kernel void Op3dTensorGeneric(global MIOPEN_TYPE* a,
                                const int a_nstride,
                                const int a_cstride,
                                global MIOPEN_TYPE* b,
                                const int b_c,
                                const int b_h,
                                const int b_nstride,
                                const int b_cstride,
                                global MIOPEN_TYPE* c,
                                const int c_c,
                                const int c_h,
                                const int c_nstride,
                                const int c_cstride,
                                const float alpha0,
                                const float alpha1,
                                const float beta,
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
    for(; gid < num_wg; gid += MAX_NUM_WG)
    {

        int lid = get_local_id(0);
        // MIOPEN_TYPE operand = b[gid + Boffset];
        int o_c_div = bitmap & (1 << 0) ? 1 : c_h;
        int o_n_div = o_c_div * (bitmap & (1 << 1) ? 1 : c_c);

        int o_h_gid_off = gid % b_h;
        int o_c_gid_off = (gid / b_h) % b_c;
        int o_n_gid_off = (gid / b_h) / b_c;

        int bindex          = o_n_gid_off * b_nstride + o_c_gid_off * b_cstride + o_h_gid_off;
        MIOPEN_TYPE operand = b_off[bindex] * alpha1;

        while(lid < work_per_wg)
        {
            int o_h = (bitmap & (1 << 0)) ? o_h_gid_off : lid % c_h;
            int o_c = (bitmap & (1 << 1)) ? o_c_gid_off : (lid / o_c_div) % c_c;
            int o_n = (bitmap & (1 << 2)) ? o_n_gid_off : lid / o_n_div;

            int aindex = o_n * a_nstride + o_c * a_cstride + o_h;
            int cindex = o_n * c_nstride + o_c * c_cstride + o_h;

            c_off[cindex] =
                MIOPEN_TENSOR_OP(a_off[aindex] * alpha0, operand) + beta * c_off[cindex];

            lid += get_local_size(0);
        }
    }
}

// NC
__kernel void Op2dTensorGeneric(global MIOPEN_TYPE* a,
                                const int a_nstride,
                                global MIOPEN_TYPE* b,
                                const int b_c,
                                const int b_nstride,
                                global MIOPEN_TYPE* c,
                                const int c_c,
                                const int c_nstride,
                                const float alpha0,
                                const float alpha1,
                                const float beta,
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

    int o_n_div = bitmap & (1 << 0) ? 1 : c_c;

    // num_wg: the number of workgroups should be launched
    // MAX_NUM_WG: the maximum number of workgroups actually launched
    for(; gid < num_wg; gid += MAX_NUM_WG)
    {

        int lid         = get_local_id(0);
        int o_c_gid_off = gid % b_c;
        int o_n_gid_off = gid / b_c;

        int bindex          = o_n_gid_off * b_nstride + o_c_gid_off;
        MIOPEN_TYPE operand = b_off[bindex] * alpha1;

        while(lid < work_per_wg)
        {
            int o_c    = (bitmap & (1 << 0)) ? o_c_gid_off : lid % c_c;
            int o_n    = (bitmap & (1 << 1)) ? o_n_gid_off : lid / o_n_div;
            int aindex = o_n * a_nstride + o_c;
            int cindex = o_n * c_nstride + o_c;
            c_off[cindex] =
                MIOPEN_TENSOR_OP(a_off[aindex] * alpha0, operand) + beta * c_off[cindex];
            lid += get_local_size(0);
        }
    }
}

// N
__kernel void Op1dTensorGeneric(global MIOPEN_TYPE* a,
                                global MIOPEN_TYPE* b,
                                const int b_n,
                                global MIOPEN_TYPE* c,
                                const int c_n,
                                const float alpha0,
                                const float alpha1,
                                const float beta,
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
    for(; gid < num_wg; gid += MAX_NUM_WG)
    {
        int lid             = get_local_id(0);
        int o_n_gid_off     = gid % b_n;
        int bindex          = o_n_gid_off;
        MIOPEN_TYPE operand = b_off[bindex] * alpha1;
        while(lid < work_per_wg)
        {
            int o_n    = (bitmap & (1 << 0)) ? o_n_gid_off : lid % c_n;
            c_off[o_n] = MIOPEN_TENSOR_OP(a_off[o_n] * alpha0, operand) + beta * c_off[o_n];
            lid += get_local_size(0);
        }
    }
}
