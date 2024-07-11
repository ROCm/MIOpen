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
#define _FLOAT_PREC float
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

template <typename TIO, typename TPREC>
__forceinline__ __device__ void bn_fwd_infer_sparial(const TIO* __restrict in, /* x input */
                                                     TIO* __restrict out,      /* y output */
                                                     const TPREC* __restrict estimatedMean,
                                                     const TPREC* __restrict estimatedVariance,
                                                     const TPREC* __restrict scale,
                                                     const TPREC* __restrict bias,
                                                     double epsilon,
                                                     unsigned int batchSize,
                                                     unsigned int imageDims,
                                                     unsigned int batchStride)
{

    size_t xidx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t yidx = blockIdx.y * blockDim.y + threadIdx.y;

    const TPREC mean        = estimatedMean[xidx];
    const TPREC variance    = estimatedVariance[xidx];
    const TPREC pscale      = scale[xidx];
    const TPREC pbias       = bias[xidx];
    const TPREC invVariance = static_cast<TPREC>(rsqrt(fabs(variance + epsilon)));

    for(size_t idx = yidx; idx < imageDims; idx += gridDim.y * blockDim.y)
    {
        for(size_t n = 0; n < batchSize; n++)
        {
            size_t index = (n * batchStride) + (xidx * imageDims) + idx;
            TPREC inhat  = (static_cast<TPREC>(in[index]) - mean) * invVariance;
            out[index]   = static_cast<TIO>(fma(pscale, inhat, pbias));
        }
    }
}

extern "C" __global__ void __launch_bounds__(MIO_BN_GRP0* MIO_BN_GRP1* MIO_BN_GRP2)
    MIOpenBatchNormFwdInferSpatialEstHIP(const _FLOAT* __restrict in, /* x input */
                                         _FLOAT* __restrict out,      /* y output */
                                         const _FLOAT_PREC* __restrict estimatedMean,
                                         const _FLOAT_PREC* __restrict estimatedVariance,
                                         const _FLOAT_PREC* __restrict scale,
                                         const _FLOAT_PREC* __restrict bias,
                                         double epsilon,
                                         unsigned int batchSize,
                                         unsigned int imageDims,
                                         unsigned int batchStride)
{
    bn_fwd_infer_sparial<_FLOAT, _FLOAT_PREC>(in,
                                              out,
                                              estimatedMean,
                                              estimatedVariance,
                                              scale,
                                              bias,
                                              epsilon,
                                              batchSize,
                                              imageDims,
                                              batchStride);
}
