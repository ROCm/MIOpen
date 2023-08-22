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
#ifdef MIOPEN_BETA_API
#define MIOPEN_TYPE float

#define GET(buf, idx) buf[idx]
#define SET(buf, idx, val) buf[idx] = val

#define GET_VAL(x, idx) GET(x, idx)
#define SET_VAL(x, idx, val) SET(x, idx, val)

#if MIOPEN_USE_FP64 == 1
#define FSTYPE double
#else
#define FSTYPE float
#endif

#define LOCAL_SIZE 256

extern "C" __global__ void LayernormFwdContiguous(const MIOPEN_TYPE* __restrict__ x,
                                                  MIOPEN_TYPE* __restrict__ y,
                                                  const MIOPEN_TYPE* __restrict__ weight,
                                                  const MIOPEN_TYPE* __restrict__ bias,
                                                  MIOPEN_TYPE* __restrict__ mean,
                                                  MIOPEN_TYPE* __restrict__ rstd,
                                                  float eps,
                                                  uint64_t inner_size,
                                                  bool mode)
{
    /*
     * Each group works on a single channel.
     * Example)
     * x dim = {N, C, L}, normalized shape = {C, L}
     * outer_size = N, inner_size = C * L
     *
     * Example2)
     * x dim = {N, C, L}, normalized shape = {L}
     * outer_size = N * C, inner_size = L
     *
     * => gws = {outer_size * LOCAL_SIZE}, lws = {LOCAL_SIZE}
     */

    /*
     * Reduction to calculate mean and rstd
     */

    const uint64_t gid = blockIdx.x;
    const uint64_t lid = threadIdx.x;

    FSTYPE pmean = 0.0f;
    FSTYPE pvar  = 0.0f;
    __shared__ FSTYPE ltmp1[LOCAL_SIZE];
    __shared__ FSTYPE ltmp2[LOCAL_SIZE];

    // reduce sum for mean and var
    for(uint64_t i = lid; i < inner_size; i += LOCAL_SIZE)
    {
        uint64_t x_idx = gid * inner_size + i;

        FSTYPE tmp = GET_VAL(x, x_idx);
        pmean += tmp;
        pvar += tmp * tmp;
    }

    ltmp1[lid] = pmean;
    ltmp2[lid] = pvar;
    __syncthreads();
    for(uint64_t i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            ltmp1[lid] += ltmp1[lid + i];
            ltmp2[lid] += ltmp2[lid + i];
        }
        __syncthreads();
    }
    pmean        = ltmp1[0] / inner_size;
    pvar         = ltmp2[0] / inner_size - pmean * pmean;
    FSTYPE prstd = rsqrt(pvar + eps);

    if(lid == 0)
    {
        if(mean)
            SET_VAL(mean, gid, pmean);
        if(rstd)
            SET_VAL(rstd, gid, prstd);
    }

    // forward calculation
    for(uint64_t i = lid; i < inner_size; i += LOCAL_SIZE)
    {
        uint64_t idx = gid * inner_size + i;

        FSTYPE pweight;
        FSTYPE pbias;

        pweight = mode ? 1 : GET_VAL(weight, i);
        pbias   = mode ? 0 : GET_VAL(bias, i);

        FSTYPE val = (GET_VAL(x, idx) - pmean) * prstd * pweight + pbias;
        SET_VAL(y, idx, val);
    }
}

extern "C" __global__ void LayerNormBwdContiguous(const MIOPEN_TYPE* __restrict__ x,
                                                  const MIOPEN_TYPE* __restrict__ dy,
                                                  const MIOPEN_TYPE* __restrict__ weight,
                                                  const MIOPEN_TYPE* __restrict__ mean,
                                                  const MIOPEN_TYPE* __restrict__ rstd,
                                                  MIOPEN_TYPE* __restrict__ dx,
                                                  uint64_t inner_size,
                                                  bool mode)
{

    const uint64_t gid = blockIdx.x;
    const uint64_t lid = threadIdx.x;

    // reduce sum for sum1, sum2
    FSTYPE sum1 = 0.0f;
    FSTYPE sum2 = 0.0f;

    __shared__ FSTYPE ltmp1[LOCAL_SIZE];
    __shared__ FSTYPE ltmp2[LOCAL_SIZE];

    for(size_t i = lid; i < inner_size; i += LOCAL_SIZE)
    {
        size_t idx = gid * inner_size + i;

        FSTYPE weight_v = mode ? 1 : GET_VAL(weight, i);
        FSTYPE dy_v     = GET_VAL(dy, idx);

        sum1 += dy_v * GET_VAL(x, idx) * weight_v;
        sum2 += dy_v * weight_v;
    }

    ltmp1[lid] = sum1;
    ltmp2[lid] = sum2;
    __syncthreads();
    for(size_t i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            ltmp1[lid] += ltmp1[lid + i];
            ltmp2[lid] += ltmp2[lid + i];
        }
        __syncthreads();
    }

    FSTYPE ds = ltmp1[0];
    FSTYPE db = ltmp2[0];

    FSTYPE s = 1.0f / inner_size;

    FSTYPE mean_v = GET_VAL(mean, gid);
    FSTYPE rstd_v = GET_VAL(rstd, gid);

    FSTYPE a  = (db * mean_v - ds) * rstd_v * rstd_v * rstd_v * s;
    FSTYPE c2 = -(a * mean_v + db * rstd_v * s);

    for(size_t i = lid; i < inner_size; i += LOCAL_SIZE)
    {
        size_t idx = gid * inner_size + i;

        FSTYPE weight_v = mode ? 1 : GET_VAL(weight, i);
        FSTYPE dy_v     = GET_VAL(dy, idx);

        FSTYPE val = rstd_v * dy_v * weight_v + a * GET_VAL(x, idx) + c2;
        SET_VAL(dx, idx, val);
    }
}

extern "C" __global__ void LayernormBwdWeightBiasContiguous(const MIOPEN_TYPE* __restrict__ x,
                                                            const MIOPEN_TYPE* __restrict__ dy,
                                                            const MIOPEN_TYPE* __restrict__ mean,
                                                            const MIOPEN_TYPE* __restrict__ rstd,
                                                            MIOPEN_TYPE* __restrict__ dw,
                                                            MIOPEN_TYPE* __restrict__ db,
                                                            uint64_t outer_size,
                                                            uint64_t inner_size)
{

    const uint64_t gid = blockIdx.x;
    const uint64_t lid = threadIdx.x;

    __shared__ FSTYPE ltmp1[LOCAL_SIZE];
    __shared__ FSTYPE ltmp2[LOCAL_SIZE];

    FSTYPE sum1 = 0;
    FSTYPE sum2 = 0;

    for(size_t i = lid; i < outer_size; i += LOCAL_SIZE)
    {
        size_t idx = i * (inner_size) + gid;

        FSTYPE dy_v = GET_VAL(dy, idx);

        sum1 += dy_v * (GET_VAL(x, idx) - GET_VAL(mean, i)) * GET_VAL(rstd, i);
        sum2 += dy_v;
    }

    ltmp1[lid] = sum1;
    ltmp2[lid] = sum2;
    __syncthreads();
    for(size_t i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            ltmp1[lid] += ltmp1[lid + i];
            ltmp2[lid] += ltmp2[lid + i];
        }
        __syncthreads();
    }

    sum1 = ltmp1[0];
    sum2 = ltmp2[0];

    if(lid == 0)
    {
        if(dw)
        {
            SET_VAL(dw, gid, sum1);
        }
        if(db)
        {
            SET_VAL(db, gid, sum2);
        }
    }
}
#endif
