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
/* Steps to compute softmax:
 * 1. Compute the max per channel.
 * 2. Subtract the max from each value in the channel.
 * 3. Compute the exponent of all the values.
 * 4. Compute the sum of the vales per channel.
 * 5. Normalize based on the sum.
 *
 * We use CSR-{Vector / Stream} apprach to pick an algorithm depending on the
 * number of channels each workgroup has to work with.
 * J. L. Greathouse, M. Daga, Efficient sparse matrix-vector multiplication
 * on GPUs using the CSR storage format, in: Proc. Int'l Conf. High Performance
 * Computing, Networking, Storage and Analysis (SC'14)
*/

kernel void SoftmaxForward(global float* y, const int c, const int grid_size, const int spatial_dim)
{
#if NUM_BATCH == 1 // CSR-Vector like appraoch

    /* Entire workgroup works on one spatial_dim.
     * We use logarthmic reductions to compute max and sum per channel.
     * This approach reads in the same data thrice from DRAM but is still better
     * than launching three different kernels.
     * The workgroup begins by computing the nth image and s (spatial_dim) it
     * is working on and iterates over the entire grid until finished.
     */

    local float l_helper[256];

    int gid = get_group_id(0);
    int lid = get_local_id(0);

    // Total number of workgroups launched can be less than the gridsize, hence iterate over.
    for(gid = get_group_id(0); gid < grid_size; gid += get_num_groups(0))
    {

        int n = gid / spatial_dim; // nth image
        int s = gid % spatial_dim; // spatial dimension (h*w)

        l_helper[lid] = -FLT_MAX;

        float t_helper = -FLT_MAX; // thread_local helper var

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

        float channel_max = l_helper[0];
        t_helper          = 0.;

        // Subtract channel_max from each value
        for(int i = lid; i < c; i += get_local_size(0))
        {
            float value = y[mad24(n, c, i) * spatial_dim + s];

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

        float channel_sum = l_helper[0];

        // Normalize each value in the channel by the channel_sum
        for(int i = lid; i < c; i += get_local_size(0))
        {
            float value = y[mad24(n, c, i) * spatial_dim + s];

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

    local float l_helper[256];

    int gid = get_group_id(0);
    int lid = get_local_id(0);

    // ID of the thread within the batch
    int batch_lid = lid & (BATCH_SIZE - 1); // thread specific channel_st
    int batch     = lid / BATCH_SIZE;       // which spatial_dim or pixel

    // Batch specific n and s
    int batch_n = (NUM_BATCH * gid + batch) / spatial_dim; // nth image
    int batch_s = (NUM_BATCH * gid + batch) % spatial_dim; // which spatial_dim/pixel

    l_helper[lid] = -FLT_MAX;

    float t_helper = -FLT_MAX; // thread_local helper var

    // stores all the values touched by one thread so that we do not have load
    // again as the CSR-Vector approach
    float value[U_BATCH_SIZE];
    for(int i = 0; i < U_BATCH_SIZE; i++)
    {
        value[i] = -FLT_MAX;
    }

    // Compute max per channel
    // BATCH_SIZE threads iterate over the channels
    for(int i = batch_lid; i < c; i += BATCH_SIZE)
    {
        if(mad24(batch_n, c, i) * spatial_dim + batch_s < c * grid_size)
            value[i / BATCH_SIZE] = y[mad24(batch_n, c, i) * spatial_dim + batch_s];
        t_helper                  = max(value[i / BATCH_SIZE], t_helper);
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

    float channel_max = l_helper[batch * BATCH_SIZE];
    t_helper          = 0.;

    // Subtract channel_max from each value
    for(int i = batch_lid; i < c; i += BATCH_SIZE)
    {

        // Compute exponent of each value
        // Then sum all the values touched by this thread
        t_helper += exp(value[i / BATCH_SIZE] - channel_max);
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

    float channel_sum = l_helper[batch * BATCH_SIZE];

    // Normalize each value in the channel by the channel_sum
    for(int i = batch_lid; i < c; i += BATCH_SIZE)
    {
        value[i / BATCH_SIZE] = exp(value[i / BATCH_SIZE] - channel_max);

        if(mad24(batch_n, c, i) * spatial_dim + batch_s < c * grid_size)
            y[mad24(batch_n, c, i) * spatial_dim + batch_s] = value[i / BATCH_SIZE] / channel_sum;
    }

#endif // CSR-Vector vs CSR-Stream
}

kernel void SoftmaxBackward(
    global float* y, global float* dx, const int c, const int grid_size, const int spatial_dim)
{

#if NUM_BATCH == 1 // CSR-Vector like appraoch
    local float l_helper[256];

    int gid = get_group_id(0);
    int lid = get_local_id(0);

    // Total number of workgroups launched can be less than the gridsize, hence iterate over.
    for(gid = get_group_id(0); gid < grid_size; gid += get_num_groups(0))
    {

        int n = gid / spatial_dim; // nth image
        int s = gid % spatial_dim; // spatial dimension (h*w)

        float channel_dot = 0.f; // thread_local helper var

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
            float value = dx[mad24(n, c, i) * spatial_dim + s] - channel_dot;

            dx[mad24(n, c, i) * spatial_dim + s] = y[mad24(n, c, i) * spatial_dim + s] * value;
        }
    }

#else

    local float l_helper[256];

    int gid = get_group_id(0);
    int lid = get_local_id(0);

    // ID of the thread within the batch
    int batch_lid = lid & (BATCH_SIZE - 1);
    int batch     = lid / BATCH_SIZE;

    // Batch specific n and s
    int batch_n = (NUM_BATCH * gid + batch) / spatial_dim;
    int batch_s = (NUM_BATCH * gid + batch) % spatial_dim;

    float channel_dot = 0.f; // thread_local helper var

    // stores all the values touched by one thread so that we do not have load
    // again as the CSR-Vector approach
    float y_value[U_BATCH_SIZE];
    float dx_value[U_BATCH_SIZE];

    // Compute dot product per channel
    for(int i = batch_lid; i < c; i += BATCH_SIZE)
    {
        if(mad24(batch_n, c, i) * spatial_dim + batch_s < c * grid_size)
        {
            y_value[i / BATCH_SIZE]  = y[mad24(batch_n, c, i) * spatial_dim + batch_s];
            dx_value[i / BATCH_SIZE] = dx[mad24(batch_n, c, i) * spatial_dim + batch_s];
        }
        channel_dot += y_value[i / BATCH_SIZE] * dx_value[i / BATCH_SIZE];
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
    for(int i = batch_lid; i < c; i += BATCH_SIZE)
    {
        dx_value[i / BATCH_SIZE] -= channel_dot;

        if(mad24(batch_n, c, i) * spatial_dim + batch_s < c * grid_size)
            dx[mad24(batch_n, c, i) * spatial_dim + batch_s] =
                y_value[i / BATCH_SIZE] * dx_value[i / BATCH_SIZE];
    }

#endif // CSR-Vector vs CSR-Stream
}
