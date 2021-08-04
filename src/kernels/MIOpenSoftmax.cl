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
#define PPCAT_NX(A, B) A##B
#define PPCAT(A, B) PPCAT_NX(A, B)
#define TWO 2
#define FOUR 4
#define EIGHT 8

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

#define _FLOAT2 PPCAT(_FLOAT, TWO)
#define _FLOAT4 PPCAT(_FLOAT, FOUR)
#define _FLOAT8 PPCAT(_FLOAT, EIGHT)

#define UNUSED __attribute__((__unused__))

#ifndef NEGATIVE_CUTOFF_VAL
#if MIOPEN_USE_FP16 == 1
#define NEGATIVE_CUTOFF_VAL ((_FLOAT)(-1e4))
#else
#define NEGATIVE_CUTOFF_VAL ((_FLOAT)(-1e20))
#endif
#endif

#ifndef USE_SOFTMAX_LOG
#define USE_SOFTMAX_LOG 0
#endif

#ifndef USE_SOFTMAX_ACCURATE
#define USE_SOFTMAX_ACCURATE 0
#endif

#ifndef USE_SOFTMAX_FAST
#define USE_SOFTMAX_FAST 0
#endif

#ifndef USE_SOFTMAX_MODE_INSTANCE
#define USE_SOFTMAX_MODE_INSTANCE 0
#endif

#ifndef USE_SOFTMAX_MODE_CHANNEL
#define USE_SOFTMAX_MODE_CHANNEL 0
#endif

#ifndef USE_ALPHA
#define USE_ALPHA 0
#endif

#ifndef USE_BETA
#define USE_BETA 0
#endif

#ifndef IS_INPUT_PACKED
#define IS_INPUT_PACKED 1
#endif

#ifndef IS_OUTPUT_PACKED
#define IS_OUTPUT_PACKED 1
#endif

#ifndef IS_DINPUT_PACKED
#define IS_DINPUT_PACKED 1
#endif

#ifndef IS_DOUTPUT_PACKED
#define IS_DOUTPUT_PACKED 1
#endif

#if(USE_SOFTMAX_LOG && USE_SOFTMAX_ACCURATE) || (USE_SOFTMAX_LOG && USE_SOFTMAX_FAST) || \
    (USE_SOFTMAX_ACCURATE && USE_SOFTMAX_FAST) ||                                        \
    !(USE_SOFTMAX_LOG || USE_SOFTMAX_ACCURATE || USE_SOFTMAX_FAST)
#error "Wrong values of USE_SOFTMAX_... macros -- exactly one should be 1, others shall be 0"
#endif

#if USE_SOFTMAX_MODE_INSTANCE == USE_SOFTMAX_MODE_CHANNEL
#error "Wrong values of USE_SOFTMAX_MODE_... macros -- exactly one should be 1, others shall be 0"
#endif

typedef union GPtr
{
    _FLOAT* f;
    _FLOAT2* fv;
} GPtr;

static inline _FLOAT LogAddExp(const _FLOAT x, const _FLOAT y)
{
    _FLOAT a = max(x, y);
    if(a <= NEGATIVE_CUTOFF_VAL)
        return NEGATIVE_CUTOFF_VAL;

    _FLOAT b = min(x, y);
    if(b <= NEGATIVE_CUTOFF_VAL)
        return a;

    _FLOAT c = b - a;

    return c <= NEGATIVE_CUTOFF_VAL ? a : (a + log(exp(c) + 1));
}

/* Steps to compute softmax:
 * 1. Compute the max per channel.
 * 2. Subtract the max from each value in the channel.
 * 3. Compute the exponent of all the values.
 * 4. Compute the sum of the vales per channel.
 * 5. Normalize based on the sum.
 *
 * We use CSR-{Vector / Stream} approach to pick an algorithm depending on the
 * number of channels each workgroup has to work with.
 * J. L. Greathouse, M. Daga, Efficient sparse matrix-vector multiplication
 * on GPUs using the CSR storage format, in: Proc. Int'l Conf. High Performance
 * Computing, Networking, Storage and Analysis (SC'14)
 */

#if RUN_FORWARD
__kernel void SoftmaxForward(global _FLOAT* x,
                             global _FLOAT* y,
                             const int vector_size,
                             const int grid_size,
                             const int spatial_dim,
#if IS_INPUT_PACKED && IS_OUTPUT_PACKED
                             UNUSED
#endif
                             const int input_h,
#if IS_INPUT_PACKED && IS_OUTPUT_PACKED
                             UNUSED
#endif
                             const int input_w,
#if IS_INPUT_PACKED
                             UNUSED
#endif
                             const int in_nstr,
#if IS_INPUT_PACKED
                             UNUSED
#endif
                             const int in_cstr,
#if IS_INPUT_PACKED
                             UNUSED
#endif
                             const int in_hstr,
#if IS_OUTPUT_PACKED
                             UNUSED
#endif
                             const int out_nstr,
#if IS_OUTPUT_PACKED
                             UNUSED
#endif
                             const int out_cstr,
#if IS_OUTPUT_PACKED
                             UNUSED
#endif
                             const int out_hstr,
                             const int x_offset,
                             const int y_offset,
#if !USE_ALPHA
                             UNUSED
#endif
                             const float alpha,
#if !USE_BETA
                             UNUSED
#endif
                             const float beta)
{
#if NUM_BATCH == 1 // CSR-Vector like approach

    /* Entire workgroup works on one spatial_dim.
     * We use logarthmic reductions to compute max and sum per channel.
     * This approach reads in the same data thrice from DRAM but is still better
     * than launching three different kernels.
     * The workgroup begins by computing the nth image and s (spatial_dim) it
     * is working on and iterates over the entire grid until finished.
     */

    local _FLOAT l_helper[256];

    int gid = get_group_id(0);
    int lid = get_local_id(0);

    // Total number of workgroups launched can be less than the gridsize, hence iterate over.
    for(gid = get_group_id(0); gid < grid_size; gid += get_num_groups(0))
    {

        int n = gid / spatial_dim; // nth image
        int s = gid % spatial_dim; // spatial dimension (h*w)
#if(!IS_INPUT_PACKED || !IS_OUTPUT_PACKED) && USE_SOFTMAX_MODE_CHANNEL
        int s0 = s / input_w;
        int s1 = s % input_w;
#endif

#if !USE_SOFTMAX_FAST
        l_helper[lid] = (_FLOAT)-MAX_VAL;

        _FLOAT t_helper = (_FLOAT)-MAX_VAL; // thread_local helper var

        // Compute max per channel
        // Iterate over all the channels one thread is supposed to loop over
        // and compute max
        for(int i = lid; i < vector_size; i += get_local_size(0))
        {
#if !IS_INPUT_PACKED && USE_SOFTMAX_MODE_INSTANCE
            int i0 = i / (input_w * input_h);
            int i1 = (i % (input_w * input_h)) / input_w;
            int i2 = (i % (input_w * input_h)) % input_w;
#endif

            int x_gidx = x_offset;
#if IS_INPUT_PACKED
            x_gidx += mad24(n, vector_size, i) * spatial_dim + s;
#else
            x_gidx += n * in_nstr;
#if USE_SOFTMAX_MODE_INSTANCE
            x_gidx += i0 * in_cstr + i1 * in_hstr + i2;
#else
            x_gidx += i * in_cstr + s0 * in_hstr + s1;
#endif
#endif

            t_helper = max(x[x_gidx], t_helper);
        }

        // Now we have to compute the max from 256 values (one per each thread)
        l_helper[lid] = t_helper;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Logarithmic reduction to compute the max.
        for(int i = (get_local_size(0) >> 1); i > 0; i >>= 1)
        {
            if(lid < i)
            {
                l_helper[lid] = max(l_helper[lid], l_helper[lid + i]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        _FLOAT channel_max = l_helper[0];
#else
        _FLOAT
#endif
        t_helper =
#if USE_SOFTMAX_LOG
            NEGATIVE_CUTOFF_VAL
#else
            (_FLOAT)0.
#endif
            ;

        // Subtract channel_max from each value
        for(int i = lid; i < vector_size; i += get_local_size(0))
        {
#if !IS_INPUT_PACKED && USE_SOFTMAX_MODE_INSTANCE
            int i0 = i / (input_w * input_h);
            int i1 = (i % (input_w * input_h)) / input_w;
            int i2 = (i % (input_w * input_h)) % input_w;
#endif

            int x_gidx = x_offset;
#if IS_INPUT_PACKED
            x_gidx += mad24(n, vector_size, i) * spatial_dim + s;
#else
            x_gidx += n * in_nstr;
#if USE_SOFTMAX_MODE_INSTANCE
            x_gidx += i0 * in_cstr + i1 * in_hstr + i2;
#else
            x_gidx += i * in_cstr + s0 * in_hstr + s1;
#endif
#endif

            _FLOAT value = x[x_gidx];

            // Compute exponent of each value
            // Then sum all the values touched by this thread
            t_helper
#if USE_SOFTMAX_LOG
                = LogAddExp(value - channel_max, t_helper)
#elif USE_SOFTMAX_FAST
                += exp(value)
#else
                += exp(value - channel_max)
#endif
                ;
        }

        l_helper[lid] = t_helper;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute sum of 256 values (one for each thread)
        // Logarithmic reduction to compute the sum
        for(int i = (get_local_size(0) >> 1); i > 0; i >>= 1)
        {
            if(lid < i)
            {
                l_helper[lid]
#if USE_SOFTMAX_LOG
                    = LogAddExp(l_helper[lid], l_helper[lid + i])
#else
                    += l_helper[lid + i]
#endif
                    ;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        _FLOAT channel_sum = l_helper[0];

        // Normalize each value in the channel by the channel_sum
        for(int i = lid; i < vector_size; i += get_local_size(0))
        {
#if !IS_INPUT_PACKED && USE_SOFTMAX_MODE_INSTANCE
            int i0 = i / (input_w * input_h);
            int i1 = (i % (input_w * input_h)) / input_w;
            int i2 = (i % (input_w * input_h)) % input_w;
#endif

            int x_gidx = x_offset;
#if IS_INPUT_PACKED
            x_gidx += mad24(n, vector_size, i) * spatial_dim + s;
#else
            x_gidx += n * in_nstr;
#if USE_SOFTMAX_MODE_INSTANCE
            x_gidx += i0 * in_cstr + i1 * in_hstr + i2;
#else
            x_gidx += i * in_cstr + s0 * in_hstr + s1;
#endif
#endif

            _FLOAT value = x[x_gidx];

// Subtracting max again because we do not write the output of
// value-max to DRAM above. Doing a subtraction again is much
// faster than writing uncoalesced to DRAM
#if !USE_SOFTMAX_FAST
            value = value - channel_max;
#endif
#if !USE_SOFTMAX_LOG
            value = exp(value);
#endif

            int y_gidx = y_offset;
#if IS_OUTPUT_PACKED
            y_gidx += mad24(n, vector_size, i) * spatial_dim + s;
#else
            y_gidx += n * out_nstr;
#if USE_SOFTMAX_MODE_INSTANCE
            y_gidx += i0 * out_cstr + i1 * out_hstr + i2;
#else
            y_gidx += i * out_cstr + s0 * out_hstr + s1;
#endif
#endif

#if USE_SOFTMAX_LOG
            value -= channel_sum;
#else
            value /= channel_sum;
#endif

#if USE_ALPHA
            value *= ((_FLOAT)alpha);
#endif

#if USE_BETA
            value += y[y_gidx] * ((_FLOAT)beta);
#endif

            y[y_gidx] = value;
        }
        (void)s;
    }

#else // CSR-Stream like approach

    /* Each workgroup is computing the softmax for NUM_BATCH spatial_dims ala CSR-Stream.
     * The number of threads iterting over channels to compute softmax for one batch is BATCH_SIZE.
     * The number of values each thread works on is U_BATCH_SIZE (read micro batch size).
     * Each batch in the workgroup works on its nth image and s (spatial_dim).
     * E.g. a 256 thread workgroup with c=31 has 8 batches and a batchsize of 32.
     * The number of workgroups launched are exactly the number as required
     * hence, there is no for-loop.
     */

    local _FLOAT l_helper[256];

    int gid = get_group_id(0);
    int lid = get_local_id(0);

    // ID of the thread within the batch
    int batch_lid = lid & (BATCH_SIZE - 1); // thread specific channel_st
    int batch     = lid / BATCH_SIZE;       // which spatial_dim or pixel

    // Batch specific n and s
    int batch_n   = (NUM_BATCH * gid + batch) / spatial_dim; // nth image
    int batch_s   = (NUM_BATCH * gid + batch) % spatial_dim; // which spatial_dim/pixel
#if(!IS_INPUT_PACKED || !IS_OUTPUT_PACKED) && USE_SOFTMAX_MODE_CHANNEL
    int batch_s0  = batch_s / input_w;
    int batch_s1  = batch_s % input_w;
#endif
#if !USE_SOFTMAX_FAST
    l_helper[lid] = (_FLOAT)-MAX_VAL;

    _FLOAT t_helper = (_FLOAT)-MAX_VAL; // thread_local helper var
#endif
// stores all the values touched by one thread so that we do not have load
// again as the CSR-Vector approach

// Comment1: Local memory is used for fp16 to get around the compiler issue reported
// in rocm2.0 in SWDEV-175176 JIRA ticket
#if MIOPEN_USE_FP16 == 1
    local _FLOAT values[U_BATCH_SIZE * 256];
#else
    _FLOAT values[U_BATCH_SIZE];
    for(int i = 0; i < U_BATCH_SIZE; i++)
    {
        values[i] = (_FLOAT)(-MAX_VAL);
    }
#endif

    // Compute max per channel
    // BATCH_SIZE threads iterate over the channels
    int index0 = batch_lid / BATCH_SIZE;
    int index  = index0;
    for(int i = batch_lid; i < vector_size; i += BATCH_SIZE, index++)
    {
        if(mad24(batch_n, vector_size, i) * spatial_dim + batch_s < vector_size * grid_size)
        {
#if !IS_INPUT_PACKED && USE_SOFTMAX_MODE_INSTANCE
            int i0 = i / (input_w * input_h);
            int i1 = (i % (input_w * input_h)) / input_w;
            int i2 = (i % (input_w * input_h)) % input_w;
#endif

            int x_gidx = x_offset;
#if IS_INPUT_PACKED
            x_gidx += mad24(batch_n, vector_size, i) * spatial_dim + batch_s;
#else
            x_gidx += batch_n * in_nstr;
#if USE_SOFTMAX_MODE_INSTANCE
            x_gidx += i0 * in_cstr + i1 * in_hstr + i2;
#else
            x_gidx += i * in_cstr + batch_s0 * in_hstr + batch_s1;
#endif
#endif

#if MIOPEN_USE_FP16 == 1 // Refer to Comment1 above for different fp16 and fp32 impl
            _FLOAT tmp                         = x[x_gidx];
#if !USE_SOFTMAX_FAST
            t_helper                           = max(tmp, t_helper);
#endif
            values[lid * U_BATCH_SIZE + index] = tmp;
#else
            values[index] = x[x_gidx];
#if !USE_SOFTMAX_FAST
            t_helper      = max(values[index], t_helper);
#endif
#endif
        }
    }

#if !USE_SOFTMAX_FAST
    // Now we have to compute the max from 256 values (one per each thread)
    l_helper[lid] = t_helper;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Logarithmic reduction to compute the max.
    for(int i = (BATCH_SIZE >> 1); i > 0; i >>= 1)
    {
        if(batch_lid < i)
        {
            l_helper[lid] = max(l_helper[lid], l_helper[lid + i]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    _FLOAT channel_max = l_helper[batch * BATCH_SIZE];
#else
    _FLOAT
#endif
    t_helper =
#if USE_SOFTMAX_LOG
        NEGATIVE_CUTOFF_VAL
#else
        (_FLOAT)0.
#endif
        ;

    barrier(CLK_LOCAL_MEM_FENCE);
    // Subtract channel_max from each value
    index = index0;
    for(int i = batch_lid; i < vector_size; i += BATCH_SIZE, index++)
    {
        // Compute exponent of each value
        // Then sum all the values touched by this thread
        int v_idx = index;
#if MIOPEN_USE_FP16 == 1
        v_idx += lid * U_BATCH_SIZE;
#endif

        _FLOAT tmp = values[v_idx];
#if !USE_SOFTMAX_FAST
        tmp -= channel_max;
#endif

#if !USE_SOFTMAX_LOG
        tmp      = exp(tmp);
#endif

#if USE_SOFTMAX_LOG
        t_helper = LogAddExp(t_helper, tmp);
#else
        t_helper += tmp;
#endif

        values[v_idx] = tmp;
    }

    l_helper[lid] = t_helper;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute sum of 256 values (one for each thread)
    // Logarithmic reduction to compute the sum
    for(int i = (BATCH_SIZE >> 1); i > 0; i >>= 1)
    {
        if(batch_lid < i)
        {
            l_helper[lid]
#if USE_SOFTMAX_LOG
                = LogAddExp(l_helper[lid], l_helper[lid + i])
#else
                += l_helper[lid + i]
#endif
                ;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    _FLOAT channel_sum = l_helper[batch * BATCH_SIZE];

    // Normalize each value in the channel by the channel_sum
    index = index0;
    for(int i = batch_lid; i < vector_size; i += BATCH_SIZE, index++)
    {
        if(mad24(batch_n, vector_size, i) * spatial_dim + batch_s < vector_size * grid_size)
        {
#if !IS_INPUT_PACKED && USE_SOFTMAX_MODE_INSTANCE
            int i0 = i / (input_w * input_h);
            int i1 = (i % (input_w * input_h)) / input_w;
            int i2 = (i % (input_w * input_h)) % input_w;
#endif

            int y_gidx = y_offset;
#if IS_OUTPUT_PACKED
            y_gidx += mad24(batch_n, vector_size, i) * spatial_dim + batch_s;
#else
            y_gidx += batch_n * out_nstr;
#if USE_SOFTMAX_MODE_INSTANCE
            y_gidx += i0 * out_cstr + i1 * out_hstr + i2;
#else
            y_gidx += i * out_cstr + batch_s0 * out_hstr + batch_s1;
#endif
#endif

            int v_idx = index;
#if MIOPEN_USE_FP16 == 1
            v_idx += lid * U_BATCH_SIZE;
#endif

#if USE_SOFTMAX_LOG
            values[v_idx] -= channel_sum;
#else
            values[v_idx] /= channel_sum;
#endif

#if USE_ALPHA
            values[v_idx] *= ((_FLOAT)alpha);
#endif

#if USE_BETA
            values[v_idx] += y[y_gidx] * ((_FLOAT)beta);
#endif

            y[y_gidx] = values[v_idx];
        }
    }

#endif // CSR-Vector vs CSR-Stream
}
#endif

#if !RUN_FORWARD
__kernel void SoftmaxBackward(global _FLOAT* y,
                              global _FLOAT* dy,
                              global _FLOAT* dx,
                              const int vector_size,
                              const int grid_size,
                              const int spatial_dim,
#if IS_OUTPUT_PACKED && IS_DOUTPUT_PACKED && IS_DINPUT_PACKED
                              UNUSED
#endif
                              const int input_h,
#if IS_OUTPUT_PACKED && IS_DOUTPUT_PACKED && IS_DINPUT_PACKED
                              UNUSED
#endif
                              const int input_w,
#if IS_OUTPUT_PACKED
                              UNUSED
#endif
                              const int out_nstr,
#if IS_OUTPUT_PACKED
                              UNUSED
#endif
                              const int out_cstr,
#if IS_OUTPUT_PACKED
                              UNUSED
#endif
                              const int out_hstr,
#if IS_DOUTPUT_PACKED
                              UNUSED
#endif
                              const int dout_nstr,
#if IS_DOUTPUT_PACKED
                              UNUSED
#endif
                              const int dout_cstr,
#if IS_DOUTPUT_PACKED
                              UNUSED
#endif
                              const int dout_hstr,
#if IS_DINPUT_PACKED
                              UNUSED
#endif
                              const int din_nstr,
#if IS_DINPUT_PACKED
                              UNUSED
#endif
                              const int din_cstr,
#if IS_DINPUT_PACKED
                              UNUSED
#endif
                              const int din_hstr,
                              const int y_offset,
                              const int dy_offset,
                              const int dx_offset,
#if !USE_ALPHA
                              UNUSED
#endif
                              const float alpha,
#if !USE_BETA
                              UNUSED
#endif
                              const float beta)
{

#if NUM_BATCH == 1 // CSR-Vector like appraoch
    local _FLOAT l_helper[256];

    int gid = get_group_id(0);
    int lid = get_local_id(0);

    // Total number of workgroups launched can be less than the gridsize, hence iterate over.
    for(gid = get_group_id(0); gid < grid_size; gid += get_num_groups(0))
    {

        int n = gid / spatial_dim; // nth image
        int s = gid % spatial_dim; // spatial dimension (h*w)
#if(!IS_DINPUT_PACKED || !IS_DOUTPUT_PACKED || !IS_OUTPUT_PACKED) && USE_SOFTMAX_MODE_CHANNEL
        int s0 = s / input_w;
        int s1 = s % input_w;
#endif

        _FLOAT channel_dot = (_FLOAT)0; // thread_local helper var

        // Compute dot product per channel
        // Iterate over all the channels one thread is supposed to loop over
        // and compute dot-product
        for(int i = lid; i < vector_size; i += get_local_size(0))
        {
#if(!IS_OUTPUT_PACKED || !IS_DOUTPUT_PACKED) && USE_SOFTMAX_MODE_INSTANCE
            int i0 = i / (input_w * input_h);
            int i1 = (i % (input_w * input_h)) / input_w;
            int i2 = (i % (input_w * input_h)) % input_w;
#endif

#if !USE_SOFTMAX_LOG
            int y_gidx = y_offset;
#if IS_OUTPUT_PACKED
            y_gidx += mad24(n, vector_size, i) * spatial_dim + s;
#else
            y_gidx += n * out_nstr;
#if USE_SOFTMAX_MODE_INSTANCE
            y_gidx += i0 * out_cstr + i1 * out_hstr + i2;
#else
            y_gidx += i * out_cstr + s0 * out_hstr + s1;
#endif
#endif
#endif

            int dy_gidx = dy_offset;
#if IS_DOUTPUT_PACKED
            dy_gidx += mad24(n, vector_size, i) * spatial_dim + s;
#else
            dy_gidx += n * dout_nstr;
#if USE_SOFTMAX_MODE_INSTANCE
            dy_gidx += i0 * dout_cstr + i1 * dout_hstr + i2;
#else
            dy_gidx += i * dout_cstr + s0 * dout_hstr + s1;
#endif
#endif

            channel_dot +=
#if !USE_SOFTMAX_LOG
                y[y_gidx] *
#endif
                dy[dy_gidx];
        }

        // Now we have to compute the sum from 256 values (one per each thread)
        l_helper[lid] = channel_dot;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Logarithmic reduction to compute the sum.
        for(int i = (get_local_size(0) >> 1); i > 0; i >>= 1)
        {
            if(lid < i)
            {
                l_helper[lid] += l_helper[lid + i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        channel_dot = l_helper[0];

        // Subtract and element-wise multiplication
        for(int i = lid; i < vector_size; i += get_local_size(0))
        {
#if((!USE_SOFTMAX_LOG && !IS_OUTPUT_PACKED) || !IS_DOUTPUT_PACKED || !IS_DINPUT_PACKED) && \
    USE_SOFTMAX_MODE_INSTANCE
            int i0 = i / (input_w * input_h);
            int i1 = (i % (input_w * input_h)) / input_w;
            int i2 = (i % (input_w * input_h)) % input_w;
#endif

            int dy_gidx = dy_offset;
#if IS_DOUTPUT_PACKED
            dy_gidx += mad24(n, vector_size, i) * spatial_dim + s;
#else
            dy_gidx += n * dout_nstr;
#if USE_SOFTMAX_MODE_INSTANCE
            dy_gidx += i0 * dout_cstr + i1 * dout_hstr + i2;
#else
            dy_gidx += i * dout_cstr + s0 * dout_hstr + s1;
#endif
#endif

            int dx_gidx = dx_offset;
#if IS_DINPUT_PACKED
            dx_gidx += mad24(n, vector_size, i) * spatial_dim + s;
#else
            dx_gidx += n * din_nstr;
#if USE_SOFTMAX_MODE_INSTANCE
            dx_gidx += i0 * din_cstr + i1 * din_hstr + i2;
#else
            dx_gidx += i * din_cstr + s0 * din_hstr + s1;
#endif
#endif

            int y_gidx = y_offset;
#if IS_OUTPUT_PACKED
            y_gidx += mad24(n, vector_size, i) * spatial_dim + s;
#else
            y_gidx += n * out_nstr;
#if USE_SOFTMAX_MODE_INSTANCE
            y_gidx += i0 * out_cstr + i1 * out_hstr + i2;
#else
            y_gidx += i * out_cstr + s0 * out_hstr + s1;
#endif
#endif

            _FLOAT value = dy[dy_gidx];
#if USE_SOFTMAX_LOG
            value -= channel_dot * exp(y[y_gidx]);
#else
            value = (value - channel_dot) * y[y_gidx];
#endif

#if USE_ALPHA
            value *= alpha;
#endif

#if USE_BETA
            value += dx[dx_gidx] * beta;
#endif

            dx[dx_gidx] = value;
        }
        (void)s;
    }

#else

    local _FLOAT l_helper[256];

    int gid = get_group_id(0);
    int lid = get_local_id(0);

    // ID of the thread within the batch
    int batch_lid = lid & (BATCH_SIZE - 1); // thread specific channel_st
    int batch     = lid / BATCH_SIZE;       // which spatial_dim or pixel

    // Batch specific n and s
    int batch_n        = (NUM_BATCH * gid + batch) / spatial_dim; // nth image
    int batch_s        = (NUM_BATCH * gid + batch) % spatial_dim; // which spatial_dim/pixel
#if(!IS_DINPUT_PACKED || !IS_DOUTPUT_PACKED || !IS_OUTPUT_PACKED) && USE_SOFTMAX_MODE_CHANNEL
    int batch_s0       = batch_s / input_w;
    int batch_s1       = batch_s % input_w;
#endif
    _FLOAT channel_dot = (_FLOAT)(0); // thread_local helper var

// stores all the values touched by one thread so that we do not have load
// again as the CSR-Vector approach
#if MIOPEN_USE_FP16 == 1
    local _FLOAT y_value[U_BATCH_SIZE * 256];
    local _FLOAT dx_value[U_BATCH_SIZE * 256];
#else
    _FLOAT y_value[U_BATCH_SIZE];
    _FLOAT dx_value[U_BATCH_SIZE];
#endif

    for(int i = 0; i < U_BATCH_SIZE; i++)
    {
#if MIOPEN_USE_FP16 == 1 // Refer to Comment1 above for different fp16 and fp32 impl
        y_value[lid * U_BATCH_SIZE + i]  = 0;
        dx_value[lid * U_BATCH_SIZE + i] = 0;
#else
        y_value[i]  = 0;
        dx_value[i] = 0;
#endif
    }

    // Compute dot product per channel
    // BATCH_SIZE threads iterate over the channels
    int index0 = batch_lid / BATCH_SIZE;
    int index  = index0;
    for(int i = batch_lid; i < vector_size; i += BATCH_SIZE, index++)
    {
        if(mad24(batch_n, vector_size, i) * spatial_dim + batch_s < vector_size * grid_size)
        {
#if(!IS_OUTPUT_PACKED || !IS_DOUTPUT_PACKED) && USE_SOFTMAX_MODE_INSTANCE
            int i0 = i / (input_w * input_h);
            int i1 = (i % (input_w * input_h)) / input_w;
            int i2 = (i % (input_w * input_h)) % input_w;
#endif

            int y_gidx = y_offset;
#if IS_OUTPUT_PACKED
            y_gidx += mad24(batch_n, vector_size, i) * spatial_dim + batch_s;
#else
            y_gidx += batch_n * out_nstr;
#if USE_SOFTMAX_MODE_INSTANCE
            y_gidx += i0 * out_cstr + i1 * out_hstr + i2;
#else
            y_gidx += i * out_cstr + batch_s0 * out_hstr + batch_s1;
#endif
#endif

            int dy_gidx = dy_offset;
#if IS_DOUTPUT_PACKED
            dy_gidx += mad24(batch_n, vector_size, i) * spatial_dim + batch_s;
#else
            dy_gidx += batch_n * dout_nstr;
#if USE_SOFTMAX_MODE_INSTANCE
            dy_gidx += i0 * dout_cstr + i1 * dout_hstr + i2;
#else
            dy_gidx += i * dout_cstr + batch_s0 * dout_hstr + batch_s1;
#endif
#endif

#if MIOPEN_USE_FP16 == 1
            _FLOAT tmp1                          = y[y_gidx];
            y_value[lid * U_BATCH_SIZE + index]  = tmp1;
            _FLOAT tmp2                          = dy[dy_gidx];
            dx_value[lid * U_BATCH_SIZE + index] = tmp2;
            channel_dot +=
#if !USE_SOFTMAX_LOG
                tmp1 *
#endif
                tmp2;
#else
            y_value[index]  = y[y_gidx];
            dx_value[index] = dy[dy_gidx];
            channel_dot +=
#if !USE_SOFTMAX_LOG
                y_value[index] *
#endif
                dx_value[index];
#endif
        }
    }

    // Now we have to compute the sum from 256 values (one per each thread)
    l_helper[lid] = channel_dot;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Logarithmic reduction to compute the sum.
    for(int i = (BATCH_SIZE >> 1); i > 0; i >>= 1)
    {
        if(batch_lid < i)
        {
            l_helper[lid] += l_helper[lid + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    channel_dot = l_helper[batch * BATCH_SIZE];

    // Subtract and element-wise multiplication
    index = index0;
    for(int i = batch_lid; i < vector_size; i += BATCH_SIZE, index++)
    {
#if !IS_DINPUT_PACKED && USE_SOFTMAX_MODE_INSTANCE
        int i0 = i / (input_w * input_h);
        int i1 = (i % (input_w * input_h)) / input_w;
        int i2 = (i % (input_w * input_h)) % input_w;
#endif

        int dx_gidx = dx_offset;
#if IS_DINPUT_PACKED
        dx_gidx += mad24(batch_n, vector_size, i) * spatial_dim + batch_s;
#else
        dx_gidx += batch_n * din_nstr;
#if USE_SOFTMAX_MODE_INSTANCE
        dx_gidx += i0 * din_cstr + i1 * din_hstr + i2;
#else
        dx_gidx += i * din_cstr + batch_s0 * din_hstr + batch_s1;
#endif
#endif

        int v_idx = index;
#if MIOPEN_USE_FP16 == 1
        v_idx += lid * U_BATCH_SIZE;
#endif

        if(mad24(batch_n, vector_size, i) * spatial_dim + batch_s < vector_size * grid_size)
        {
#if USE_SOFTMAX_LOG
            dx_value[v_idx] -= channel_dot * exp(y_value[v_idx]);
#else
            dx_value[v_idx] = (dx_value[v_idx] - channel_dot) * y_value[v_idx];
#endif

#if USE_ALPHA
            dx_value[v_idx] *= alpha;
#endif

#if USE_BETA
            dx_value[v_idx] += dx[dx_gidx] * beta;
#endif

            dx[dx_gidx] = dx_value[v_idx];
        }
    }

#endif // CSR-Vector vs CSR-Stream
}
#endif
