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

#define OP_0(a, b) (a) + (b)
#define OP_1(a, b) (a) * (b)
#define OP_2(a, b) (((a) < (b)) ? (a) : (b))
#define OP_3(a, b) (((a) > (b)) ? (a) : (b))
#define OP(op, a, b) \
    (op == 0 ? OP_0(a, b) : (op == 1 ? OP_1(a, b) : (op == 2 ? OP_2(a, b) : OP_3(a, b))))

#define UNUSED __attribute__((__unused__))

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
                              int op)
{
    int gid = get_group_id(0);
    int lid = get_local_id(0);

#if INCR_WG == 1
    int o_n             = gid / b_c;
    int o_c             = gid % b_c;
    MIOPEN_TYPE operand = b[o_c];

    while(lid < work_per_wg)
    {
        c[o_n * c_nstride + o_c * c_cstride + lid] =
            OP(op, a[o_n * c_nstride + o_c * c_cstride + lid], operand);

        lid += get_local_size(0);
    }

// each workgroup computes N*H*W for each C (bias-term)
// number of workgroups = c_c (b_c)
#elif INCR_WG == 0
    MIOPEN_TYPE operand = b[gid];
    int work_off        = work_per_wg / c_n;

    while(lid < work_per_wg)
    {
        int o_hw = lid % work_off;
        int o_n  = lid / work_off;
        c[o_n * c_nstride + gid * c_cstride + o_hw] =
            OP(op, a[o_n * c_nstride + gid * c_cstride + o_hw], operand);

        lid += get_local_size(0);
    }
#endif // INCR_WG
}

__kernel void OpTensorGeneric(global MIOPEN_TYPE* a,
                              global MIOPEN_TYPE* b,
                              const int b_c,
                              const int b_h,
                              const int b_w,
                              const int b_nstride,
                              const int b_cstride,
                              global MIOPEN_TYPE* c,
                              const int c_c,
                              const int c_h,
                              const int c_w,
                              const int c_nstride,
                              const int c_cstride,
                              const unsigned int bitmap,
                              const int work_per_wg,
                              int op)
{
    int gid = get_group_id(0);
    int lid = get_local_id(0);

#if LEADING_ONES == 1 && FIRST_NOT_ONE == 2 // bitmap = 1,1,1,0
    MIOPEN_TYPE operand = b[gid];

    int o_h = gid % c_h;
    int o_c = (gid / c_h) % c_c;
    int o_n = gid / (c_c * c_h);

    while(lid < work_per_wg)
    {
        c[o_n * c_nstride + o_c * c_cstride + o_h * c_w + lid] =
            OP(op, a[o_n * c_nstride + o_c * c_cstride + o_h * c_w + lid], operand);

        lid += get_local_size(0);
    }
#elif LEADING_ONES == 1 && FIRST_NOT_ONE == 1 // bitmap = 1,1,0,0
    MIOPEN_TYPE operand = b[gid];

    int o_c = gid % c_c;
    int o_n = gid / c_c;

    while(lid < work_per_wg)
    {
        c[o_n * c_nstride + o_c * c_cstride + lid] =
            OP(op, a[o_n * c_nstride + o_c * c_cstride + lid], operand);

        lid += get_local_size(0);
    }

#elif LEADING_ONES == 1 && FIRST_NOT_ONE == 0 // bitmap = 1,0,0,0
    MIOPEN_TYPE operand = b[gid];

    while(lid < work_per_wg)
    {
        c[gid * c_nstride + lid] = OP(op, a[gid * c_nstride + lid], operand);

        lid += get_local_size(0);
    }

#else // generic op tensor
    MIOPEN_TYPE operand = b[gid];
    int o_h_div         = bitmap & (1 << 0) ? 1 : c_w;
    int o_c_div         = o_h_div * (bitmap & (1 << 1) ? 1 : c_h);
    int o_n_div         = o_c_div * (bitmap & (1 << 2) ? 1 : c_c);

    int o_w_gid_off = gid % b_w;
    int o_h_gid_off = (gid / b_w) % b_h;
    int o_c_gid_off = (gid / b_cstride) % b_c;
    int o_n_gid_off = gid / b_nstride;

    while(lid < work_per_wg)
    {
        int o_w = (bitmap & (1 << 0)) ? o_w_gid_off : lid % c_w;
        int o_h = (bitmap & (1 << 1)) ? o_h_gid_off : (lid / o_h_div) % c_h;
        int o_c = (bitmap & (1 << 2)) ? o_c_gid_off : (lid / o_c_div) % c_c;
        int o_n = (bitmap & (1 << 3)) ? o_n_gid_off : lid / o_n_div;

        c[o_n * c_nstride + o_c * c_cstride + o_h * c_w + o_w] =
            OP(op, a[o_n * c_nstride + o_c * c_cstride + o_h * c_w + o_w], operand);

        lid += get_local_size(0);
    }
#endif
}
