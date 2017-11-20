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

#define _FLOAT float
#define _FLOAT2 float2
#define _FLOAT4 float4
#define _FLOAT8 float8

#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38F /* max value */
#endif

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

// Disable specific warnings
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconditional-uninitialized"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsometimes-uninitialized"
#endif

__kernel void BatchNormBwdPerActivationSaved(const __global _FLOAT* x_in,
                                             const __global _FLOAT* dy_in,
                                             unsigned int N,
                                             unsigned int in_nstride,
                                             unsigned int in_cstride,
                                             __global _FLOAT* dx_out,
                                             const __global _FLOAT* scale,
                                             __global _FLOAT* delta_scale,
                                             __global _FLOAT* delta_bias,
                                             const __global _FLOAT* savedMean,
                                             const __global _FLOAT* savedInvVariance)
{
    /*
    for(int n = 0; n < N; n++) {
     for(int c = 0; c < C; c++) {
      for(int h = 0; h < H; h++) {
       for(int w = 0; w < W; w++) {
        float pixel_val = input_image[n*C*H*W + c*H*W + h*W +w];
    }}}}
            C*H*W is also stored as in_nstride,
            H*W is in_cstride,
            W is in_hstride.
    ------------------------------------------
    http://kratzert.github.io
    http://cthorey.github.io./backpropagation/
    ------------------------------------------
    mu = 1./N*np.sum(h, axis = 0)
    var = 1./N*np.sum((h-mu)**2, axis = 0)
    dbeta = np.sum(dy, axis=0)
    dgamma = np.sum((h - mu) * (var + eps)**(-1. / 2.) * dy, axis=0)
    dh = (1. / N) * gamma * (var + eps)**(-1. / 2.) * (N * dy - np.sum(dy, axis=0)
        - (h - mu) * (var + eps)**(-1.0) * np.sum(dy * (h - mu), axis=0))
    */
    int xgid    = get_global_id(0);
    int ygid    = get_global_id(1);
    int yglb_sz = get_global_size(1);
    int Cidx    = in_cstride * xgid;

    unsigned int inImgIndex, index, adjIndex;
    _FLOAT mean, invVar;
    _FLOAT elemStd, xhat, dyelem;
    _FLOAT pvt_scale, pvt_dscale;
    _FLOAT pvt_dbias;
    _FLOAT tmp1, tmp2, tmp3;
    _FLOAT dxhat    = 0.;
    _FLOAT dxhathat = 0.;

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
            pvt_dscale = 0.;
            pvt_dbias  = 0.;
            dxhat      = 0.;
            dxhathat   = 0.;

            for(int n = 0; n < N; n++)
            {
                // per (x-dims) channel load a block of data into LDS
                index   = in_nstride * n + adjIndex;
                elemStd = x_in[index] - mean; // (x_i - mean)
                xhat    = elemStd * invVar;
                dyelem  = dy_in[index];
                pvt_dbias += dyelem;
                pvt_dscale = mad(xhat, dyelem, pvt_dscale);
                tmp1       = pvt_scale * dyelem;
                dxhat += tmp1;
                dxhathat = mad(tmp1, xhat, dxhathat);
            } // end for(n)

            for(int n = 0; n < N; n++)
            {
                index         = in_nstride * n + adjIndex;
                elemStd       = x_in[index] - mean; // (x_i - mean)
                xhat          = elemStd * invVar;
                tmp1          = mad(xhat, dxhathat, dxhat);
                tmp2          = mad((_FLOAT)N, dxhat, -tmp1);
                tmp3          = invVar / ((_FLOAT)N);
                dx_out[index] = tmp3 * tmp2;
            }
            // Write out data
            delta_bias[adjIndex]  = pvt_dbias;
            delta_scale[adjIndex] = pvt_dscale;
        }
    } // end for(img_offset) //image mini_batch is processed
}

__kernel void BatchNormBwdPerActivation(const __global _FLOAT* x_in,
                                        const __global _FLOAT* dy_in,
                                        unsigned int N,
                                        unsigned int in_nstride,
                                        unsigned int in_cstride,
                                        __global _FLOAT* dx_out,
                                        const __global _FLOAT* scale,
                                        __global _FLOAT* delta_scale,
                                        __global _FLOAT* delta_bias,
                                        double epsilon)
{

    int xgid    = get_global_id(0);
    int ygid    = get_global_id(1);
    int yglb_sz = get_global_size(1);
    int Cidx    = in_cstride * xgid;

    unsigned int inImgIndex, index, adjIndex;
    _FLOAT mean, invVar;
    _FLOAT xhat, dyelem;
    _FLOAT pvt_scale, pvt_dscale;
    _FLOAT pvt_dbias;
    _FLOAT tmp1, tmp2, tmp3;
    _FLOAT elemStd, variance;
    _FLOAT dxhat    = 0.;
    _FLOAT dxhathat = 0.;

    // move across the sections of the image mini_batch stack
    for(int img_offset = 0; img_offset < in_cstride; img_offset += yglb_sz)
    {
        mean       = 0.;
        inImgIndex = ygid + img_offset;

        // #1 calculate the mean
        // iterating through the stack of images in the mini_batch
        if(inImgIndex < in_cstride)
        {

            adjIndex = Cidx + inImgIndex; // gamma and beta tensor index
            for(int n = 0; n < N; n++)
            {
                index = in_nstride * n + adjIndex;
                mean += x_in[index];
            } // end for(n)
            mean /= (_FLOAT)N;

            elemStd  = 0.;
            variance = 0.;
            // #2 calculate the variances
            // sigma^2 = (1/batch_mean) * sum( (x_i - batch_mean)^2 )
            for(int n = 0; n < N; n++)
            {
                // per (x-dims) channel load a block of data into LDS
                index    = in_nstride * n + adjIndex;
                elemStd  = x_in[index] - mean; // (x_i - mean) //this is reused but needs recalc
                variance = mad(elemStd, elemStd, variance); // sum{ (x_i - mean)^2 }
            }                                               // end for(n)
            variance /= (_FLOAT)N;                          // (1/N)*sum{ (x_i - mean)^2 }

            // #3 add epsilon for numeric stability, sqr_root, and invert
            invVar = rsqrt(fabs(variance + epsilon));

            pvt_scale  = scale[adjIndex];
            pvt_dscale = 0.;
            pvt_dbias  = 0.;
            dxhat      = 0.;
            dxhathat   = 0.;

            for(int n = 0; n < N; n++)
            {
                // per (x-dims) channel load a block of data into LDS
                index   = in_nstride * n + adjIndex;
                elemStd = x_in[index] - mean; // (x_i - mean)
                xhat    = elemStd * invVar;
                dyelem  = dy_in[index];
                pvt_dbias += dyelem;
                pvt_dscale = mad(xhat, dyelem, pvt_dscale);
                tmp1       = pvt_scale * dyelem;
                dxhat += tmp1;
                dxhathat = mad(tmp1, xhat, dxhathat);
            } // end for(n)

            for(int n = 0; n < N; n++)
            {
                index         = in_nstride * n + adjIndex;
                elemStd       = x_in[index] - mean; // (x_i - mean)
                xhat          = elemStd * invVar;
                tmp1          = mad(xhat, dxhathat, dxhat);
                tmp2          = mad((_FLOAT)N, dxhat, -tmp1);
                tmp3          = invVar / ((_FLOAT)N);
                dx_out[index] = tmp3 * tmp2;
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
