
/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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
#define _FLOAT half
#define _FLOAT_PREC half
#endif

#if MIOPEN_USE_FP32 == 1
#define _FLOAT float
#define _FLOAT_PREC float
#endif

// if MIOPEN_USE_FP16 is not defined, default is fp32
#ifndef _FLOAT
#define _FLOAT float
#define _FLOAT_PREC float
#endif


template <typename T>
T __device__ __forceinline__ mad(T a, T b, T c)
{
    return a * b + c;
}

extern "C"  __global__ void __launch_bounds__(MIO_BN_GRP0 * MIO_BN_GRP1 * MIO_BN_GRP2)
MIOpenBatchNormFwdInferSpatialEstHIP(const  _FLOAT* __restrict in, /* x input */
                                   _FLOAT* __restrict out,      /* y output */
                                  const  _FLOAT_PREC* __restrict estimatedMean,
                                  const  _FLOAT_PREC* __restrict estimatedVariance,
                                  const  _FLOAT_PREC* __restrict scale,
                                  const  _FLOAT_PREC* __restrict bias,
                                  double epsilon,
                                  unsigned int batchSize,
                                  unsigned int imageDims,
                                  unsigned int batchStride)
{

    int xgid = blockIdx.x * blockDim.x + threadIdx.x;
    int ygid = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int index;

    _FLOAT_PREC mean, variance, invVariance;
    _FLOAT_PREC inhat;
    _FLOAT_PREC pscale, pbias;

    mean        = *(estimatedMean + xgid);
    variance    = *(estimatedVariance + xgid);
    pscale      = *(scale + xgid);
    pbias       = *(bias + xgid);
    invVariance = rsqrt(fabs(variance + epsilon));

    for(unsigned int idx = ygid; idx < imageDims; idx += gridDim.y * blockDim.y)
    {
        for(unsigned int n = 0; n < batchSize; n++)
        {
            index      = (n * batchStride) + (xgid * imageDims) + idx;
            inhat      = ((_FLOAT_PREC)(*(in + index)) - mean) * invVariance;
            out[index] = mad<_FLOAT>(pscale, inhat, pbias);
        }
    }
} 
