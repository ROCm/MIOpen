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
#define MIO_BN_NODPP 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define _FLOAT half
#define _FLOAT_PREC half
#ifndef HALF_MAX
#define MAX_VAL 65504 /* max value */
#else
#define MAX_VAL HALF_MAX
#endif
#endif
#if MIOPEN_USE_FP32 == 1
#define _FLOAT float
#define _FLOAT_PREC float
#ifndef FLT_MAX
#define MAX_VAL 3.402823466e+38F /* max value */
#else
#define MAX_VAL FLT_MAX
#endif
#endif
#if MIOPEN_USE_FPMIX == 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define _FLOAT half
#define _FLOAT_PREC float
/*
#ifndef HALF_MAX
#define MAX_VAL 65504
#else
#define MAX_VAL HALF_MAX
#endif
*/
#endif

#define _FLOAT2 PPCAT(_FLOAT, TWO)
#define _FLOAT4 PPCAT(_FLOAT, FOUR)
#define _FLOAT8 PPCAT(_FLOAT, EIGHT)

#ifndef MIO_BN_LDS_SIZE
#define MIO_BN_LDS_SIZE 1
#endif

#ifndef MIO_BN_C
#define MIO_BN_C 1
#endif

#ifndef MIO_BN_N
#define MIO_BN_N 1
#endif

#ifndef MIO_BN_NHW
#define MIO_BN_NHW 1
#endif

#ifndef MIO_BN_CHW
#define MIO_BN_CHW 1
#endif

#ifndef MIO_BN_INHW
#define MIO_BN_INHW 1
#endif

#ifndef MIO_BN_HW
#define MIO_BN_HW 1
#endif

#ifndef MIO_BN_NODPP
#define MIO_BN_NODPP 0
#elif(MIO_BN_NODPP == 1)
#undef __AMDGCN__
#endif

// Disable specific warnings
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconditional-uninitialized"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsometimes-uninitialized"
#endif

__kernel void MIOpenBatchNormBwdPerActivationSaved(const __global _FLOAT* x_in,
                                                   const __global _FLOAT* dy_in,
                                                   unsigned int N,
                                                   unsigned int in_nstride,
                                                   unsigned int in_cstride,
                                                   __global _FLOAT* dx_out,
                                                   const __global _FLOAT_PREC* scale,
                                                   __global _FLOAT_PREC* delta_scale,
                                                   __global _FLOAT_PREC* delta_bias,
                                                   const __global _FLOAT_PREC* savedMean,
                                                   const __global _FLOAT_PREC* savedInvVariance)
{

    int xgid    = get_global_id(0);
    int ygid    = get_global_id(1);
    int yglb_sz = get_global_size(1);
    int Cidx    = in_cstride * xgid;

    unsigned int inImgIndex, index, adjIndex;
    _FLOAT_PREC mean, invVar;
    _FLOAT_PREC xhat, dyelem;
    _FLOAT_PREC pvt_scale, pvt_dscale;
    _FLOAT_PREC pvt_dbias;
    _FLOAT_PREC tmp1, tmp2, tmp3;
    _FLOAT_PREC dxhat    = (_FLOAT_PREC)0.;
    _FLOAT_PREC dxhathat = (_FLOAT_PREC)0.;

    // move across the sections of an image in the mini_batch stack
    for(int img_offset = 0; img_offset < in_cstride; img_offset += yglb_sz)
    {

        inImgIndex = img_offset + ygid;
        if(inImgIndex < in_cstride)
        {

            adjIndex   = Cidx + inImgIndex; // gamma and beta tensor index
            mean       = savedMean[adjIndex];
            invVar     = savedInvVariance[adjIndex];
            pvt_scale  = scale[adjIndex];
            pvt_dscale = (_FLOAT_PREC)0.;
            pvt_dbias  = (_FLOAT_PREC)0.;
            dxhat      = (_FLOAT_PREC)0.;
            dxhathat   = (_FLOAT_PREC)0.;

            for(int n = 0; n < N; n++)
            {
                // per (x-dims) channel load a block of data into LDS
                index  = in_nstride * n + adjIndex;
                xhat   = ((_FLOAT_PREC)(*(x_in + index)) - mean) * invVar;
                dyelem = (_FLOAT_PREC)(dy_in[index]);
                pvt_dbias += dyelem;
                pvt_dscale = mad(xhat, dyelem, pvt_dscale);
                tmp1       = pvt_scale * dyelem;
                dxhat += tmp1;
                dxhathat = mad(tmp1, xhat, dxhathat);
            } // end for(n)

            for(int n = 0; n < N; n++)
            {
                index         = in_nstride * n + adjIndex;
                xhat          = ((_FLOAT_PREC)(*(x_in + index)) - mean) * invVar;
                tmp1          = mad(xhat, dxhathat, dxhat);
                tmp2          = mad((_FLOAT_PREC)N, dxhat, -tmp1);
                tmp3          = invVar / ((_FLOAT_PREC)N);
                dx_out[index] = (_FLOAT)(tmp3 * tmp2);
            }
            // Write out data
            delta_bias[adjIndex]  = pvt_dbias;
            delta_scale[adjIndex] = pvt_dscale;
        }
    } // end for(img_offset) //image mini_batch is processed
}

__kernel void MIOpenBatchNormBwdPerActivation(const __global _FLOAT* x_in,
                                              const __global _FLOAT* dy_in,
                                              unsigned int N,
                                              unsigned int in_nstride,
                                              unsigned int in_cstride,
                                              __global _FLOAT* dx_out,
                                              const __global _FLOAT_PREC* scale,
                                              __global _FLOAT_PREC* delta_scale,
                                              __global _FLOAT_PREC* delta_bias,
                                              double epsilon)
{

    int xgid    = get_global_id(0);
    int ygid    = get_global_id(1);
    int yglb_sz = get_global_size(1);
    int Cidx    = in_cstride * xgid;

    unsigned int inImgIndex, index, adjIndex;
    _FLOAT_PREC mean, invVar;
    _FLOAT_PREC xhat, dyelem;
    _FLOAT_PREC pvt_scale, pvt_dscale;
    _FLOAT_PREC pvt_dbias;
    _FLOAT_PREC tmp1, tmp2, tmp3;
    _FLOAT_PREC variance;
    _FLOAT_PREC dxhat    = (_FLOAT_PREC)0.;
    _FLOAT_PREC dxhathat = (_FLOAT_PREC)0.;

    // move across the sections of the image mini_batch stack
    for(int img_offset = 0; img_offset < in_cstride; img_offset += yglb_sz)
    {
        mean       = (_FLOAT_PREC)0.;
        variance   = (_FLOAT_PREC)0.;
        inImgIndex = ygid + img_offset;

        // #1 calculate the mean
        // iterating through the stack of images in the mini_batch
        if(inImgIndex < in_cstride)
        {
            mean     = 0.;
            variance = 0.;
            adjIndex = Cidx + inImgIndex; // gamma and beta tensor index

#pragma unroll
            for(int n = 0; n < MIO_BN_N; n++)
            {
                index          = in_nstride * n + adjIndex;
                _FLOAT_PREC in = (_FLOAT_PREC)(*(x_in + index));
                mean += in;
                variance = mad(in, in, variance);
            } // end for(n)
            mean /= (_FLOAT_PREC)N;
            variance /= (_FLOAT_PREC)N;
            variance = mad(-mean, mean, variance);
            invVar   = rsqrt(fabs(variance + epsilon));

            pvt_scale  = *(scale + adjIndex);
            pvt_dscale = (_FLOAT_PREC)0.;
            pvt_dbias  = (_FLOAT_PREC)0.;
            dxhat      = (_FLOAT_PREC)0.;
            dxhathat   = (_FLOAT_PREC)0.;

#pragma unroll
            for(int n = 0; n < MIO_BN_N; n++)
            {
                // per (x-dims) channel load a block of data into LDS
                index  = in_nstride * n + adjIndex;
                xhat   = ((_FLOAT_PREC)(*(x_in + index)) - mean) * invVar;
                dyelem = (_FLOAT_PREC)(dy_in[index]);
                pvt_dbias += dyelem;
                pvt_dscale = mad(xhat, dyelem, pvt_dscale);
                tmp1       = pvt_scale * dyelem;
                dxhat += tmp1;
                dxhathat = mad(tmp1, xhat, dxhathat);
            } // end for(n)

#pragma unroll
            for(int n = 0; n < MIO_BN_N; n++)
            {
                index         = in_nstride * n + adjIndex;
                xhat          = ((_FLOAT_PREC)(*(x_in + index)) - mean) * invVar;
                tmp1          = mad(xhat, dxhathat, dxhat);
                tmp2          = mad((_FLOAT_PREC)N, dxhat, -tmp1);
                tmp3          = invVar / ((_FLOAT_PREC)N);
                dx_out[index] = (_FLOAT)(tmp3 * tmp2);
            }
            // Write out data
            delta_bias[adjIndex]  = pvt_dbias;
            delta_scale[adjIndex] = pvt_dscale;
        }
    } // end for(img_offset) //image mini_batch is processed
}

//================== END PER ACTIVATION ====================

// Restore warnings
#ifdef __clang__
#pragma clang diagnostic pop
#pragma clang diagnostic pop
#endif
