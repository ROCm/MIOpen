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

typedef union GPtr
{
    _FLOAT* f;
    _FLOAT2* fv;
} GPtr;

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

__kernel void
SoftmaxForward(global _FLOAT* y, const int c, const int grid_size, const int spatial_dim)
{
#if NUM_BATCH == 1 // CSR-Vector like appraoch

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

        l_helper[lid] = (_FLOAT)-MAX_VAL;

        _FLOAT t_helper = (_FLOAT)-MAX_VAL; // thread_local helper var

        // Compute max per channel
        // Iterate over all the channels one thread is supposed to loop over
        // and compute max
        for(int i = lid; i < c; i += get_local_size(0))
        {
            t_helper = max(y[mad24(n, c, i) * spatial_dim + s], t_helper);
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
        t_helper           = 0.;

        // Subtract channel_max from each value
        for(int i = lid; i < c; i += get_local_size(0))
        {
            _FLOAT value = y[mad24(n, c, i) * spatial_dim + s];

            // Compute exponent of each value
            // Then sum all the values touched by this thread
            t_helper += exp(value - channel_max);
        }

        l_helper[lid] = t_helper;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute sum of 256 values (one for each thread)
        // Logarithmic reduction to compute the sum
        for(int i = (get_local_size(0) >> 1); i > 0; i >>= 1)
        {
            if(lid < i)
            {
                l_helper[lid] += l_helper[lid + i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        _FLOAT channel_sum = l_helper[0];

        // Normalize each value in the channel by the channel_sum
        for(int i = lid; i < c; i += get_local_size(0))
        {
            _FLOAT value = y[mad24(n, c, i) * spatial_dim + s];

            // Subtracting max again because we do not write the output of
            // value-max to DRAM above. Doing a subtraction again is much
            // faster than writing uncoalesced to DRAM
            value = exp(value - channel_max);

            y[mad24(n, c, i) * spatial_dim + s] = value / channel_sum;
        }
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
    int batch_n = (NUM_BATCH * gid + batch) / spatial_dim; // nth image
    int batch_s = (NUM_BATCH * gid + batch) % spatial_dim; // which spatial_dim/pixel

    l_helper[lid] = (_FLOAT)-MAX_VAL;

    _FLOAT t_helper = (_FLOAT)-MAX_VAL; // thread_local helper var

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
    for(int i = batch_lid; i < c; i += BATCH_SIZE, index++)
    {
        if(mad24(batch_n, c, i) * spatial_dim + batch_s < c * grid_size)
        {

#if MIOPEN_USE_FP16 == 1 // Refer to Comment1 above for different fp16 and fp32 impl
            _FLOAT tmp                         = y[mad24(batch_n, c, i) * spatial_dim + batch_s];
            t_helper                           = max(tmp, t_helper);
            values[lid * U_BATCH_SIZE + index] = tmp;
#else
            values[index] = y[mad24(batch_n, c, i) * spatial_dim + batch_s];
            t_helper      = max(values[index], t_helper);
#endif
        }
    }

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
    t_helper           = (_FLOAT)0.;

    // Subtract channel_max from each value
    index = index0;
    for(int i = batch_lid; i < c; i += BATCH_SIZE, index++)
    {
// Compute exponent of each value
// Then sum all the values touched by this thread
#if MIOPEN_USE_FP16 == 1
        _FLOAT tmp = exp(values[lid * U_BATCH_SIZE + index] - channel_max);
        t_helper += tmp;
        values[lid * U_BATCH_SIZE + index] = tmp;
#else
        _FLOAT tmp        = exp(values[index] - channel_max);
        t_helper += tmp;
        values[index] = tmp;
#endif
    }

    l_helper[lid] = t_helper;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute sum of 256 values (one for each thread)
    // Logarithmic reduction to compute the sum
    for(int i = (BATCH_SIZE >> 1); i > 0; i >>= 1)
    {
        if(batch_lid < i)
        {
            l_helper[lid] += l_helper[lid + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    _FLOAT channel_sum = l_helper[batch * BATCH_SIZE];

    // Normalize each value in the channel by the channel_sum
    index = index0;
    for(int i = batch_lid; i < c; i += BATCH_SIZE, index++)
    {
        if(mad24(batch_n, c, i) * spatial_dim + batch_s < c * grid_size)
        {
#if MIOPEN_USE_FP16 == 1
            y[mad24(batch_n, c, i) * spatial_dim + batch_s] =
                values[lid * U_BATCH_SIZE + index] / channel_sum;
#else
            y[mad24(batch_n, c, i) * spatial_dim + batch_s] = values[index] / channel_sum;
#endif
        }
    }

#endif // CSR-Vector vs CSR-Stream
}

__kernel void SoftmaxBackward(
    global _FLOAT* y, global _FLOAT* dx, const int c, const int grid_size, const int spatial_dim)
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

        _FLOAT channel_dot = (_FLOAT)0; // thread_local helper var

        // Compute dot product per channel
        // Iterate over all the channels one thread is supposed to loop over
        // and compute dot-product
        for(int i = lid; i < c; i += get_local_size(0))
        {
            channel_dot +=
                (y[mad24(n, c, i) * spatial_dim + s] * dx[mad24(n, c, i) * spatial_dim + s]);
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
        for(int i = lid; i < c; i += get_local_size(0))
        {
            _FLOAT value = dx[mad24(n, c, i) * spatial_dim + s] - channel_dot;

            dx[mad24(n, c, i) * spatial_dim + s] = y[mad24(n, c, i) * spatial_dim + s] * value;
        }
    }

#else

    local _FLOAT l_helper[256];

    int gid = get_group_id(0);
    int lid = get_local_id(0);

    // ID of the thread within the batch
    int batch_lid = lid & (BATCH_SIZE - 1); // thread specific channel_st
    int batch     = lid / BATCH_SIZE;       // which spatial_dim or pixel

    // Batch specific n and s
    int batch_n = (NUM_BATCH * gid + batch) / spatial_dim; // nth image
    int batch_s = (NUM_BATCH * gid + batch) % spatial_dim; // which spatial_dim/pixel

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
        y_value[lid * U_BATCH_SIZE + i] = 0;
        dx_value[lid * U_BATCH_SIZE + i] = 0;
#else
        y_value[i]          = 0;
        dx_value[i]         = 0;
#endif
    }

    // Compute dot product per channel
    // BATCH_SIZE threads iterate over the channels
    int index0 = batch_lid / BATCH_SIZE;
    int index  = index0;
    for(int i = batch_lid; i < c; i += BATCH_SIZE, index++)
    {
        if(mad24(batch_n, c, i) * spatial_dim + batch_s < c * grid_size)
        {
#if MIOPEN_USE_FP16 == 1
            _FLOAT tmp1                          = y[mad24(batch_n, c, i) * spatial_dim + batch_s];
            y_value[lid * U_BATCH_SIZE + index]  = tmp1;
            _FLOAT tmp2                          = dx[mad24(batch_n, c, i) * spatial_dim + batch_s];
            dx_value[lid * U_BATCH_SIZE + index] = tmp2;
            channel_dot += tmp1 * tmp2;
#else
            y_value[index]  = y[mad24(batch_n, c, i) * spatial_dim + batch_s];
            dx_value[index] = dx[mad24(batch_n, c, i) * spatial_dim + batch_s];
            channel_dot += y_value[index] * dx_value[index];
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
    for(int i = batch_lid; i < c; i += BATCH_SIZE, index++)
    {
#if MIOPEN_USE_FP16 == 1
        dx_value[lid * U_BATCH_SIZE + index] -= channel_dot;
        if(mad24(batch_n, c, i) * spatial_dim + batch_s < c * grid_size)
            dx[mad24(batch_n, c, i) * spatial_dim + batch_s] =
                y_value[lid * U_BATCH_SIZE + index] * dx_value[lid * U_BATCH_SIZE + index];
#else
        dx_value[index] -= channel_dot;
        if(mad24(batch_n, c, i) * spatial_dim + batch_s < c * grid_size)
            dx[mad24(batch_n, c, i) * spatial_dim + batch_s] = y_value[index] * dx_value[index];
#endif
    }

#endif // CSR-Vector vs CSR-Stream
}
