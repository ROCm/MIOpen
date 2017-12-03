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

#ifndef MIO_BATCHNORMHOST_H_
#define MIO_BATCHNORMHOST_H_

#include <cmath>
#include <iomanip>

#define MIO_HEIRARCH_SEL 0

#if(MIO_HEIRARCH_SEL == 1)
#define MIO_BN_DIST 32
#endif

template <typename T>
int miopenBNFwdTrainPerActivationRunHost(
    /*
        T alpha,
        T beta,
    */
    int n_batchs,
    int channels,
    int height,
    int width,
    const T* in_ptr,
    double* out_ptr,
    T* scale_ptr,
    T* bias_ptr,
    double epsilon,
    bool savemeanvar,
    bool runningmeanvar,
    double* saveMean,
    double* saveInvVariance,
    double* runningMean,
    double* runningVariance,
    double expAvgFactor)
{

    // C*H*W is also stored as in_nstride, H*W is in_cstride, W is in_hstride.
    unsigned int index;
    unsigned int adjIndex;
    unsigned int in_nstride = channels * height * width;
    unsigned int in_cstride = height * width;

    double mean_accum     = 0.;
    double variance_accum = 0.;
    double elemStd        = 0.;

    int ret = 0;
    for(int cidx = 0; cidx < channels; cidx++)
    { // via channel
        mean_accum     = 0.;
        variance_accum = 0.;
        // process the batch per channel
        for(int row = 0; row < height; row++)
        { // via rows
            for(int column = 0; column < width; column++)
            { // via columns
                mean_accum = 0.;
                adjIndex   = in_cstride * cidx + width * row + column;
                for(int bidx = 0; bidx < n_batchs; bidx++)
                { // via mini_batch
                    index = in_nstride * bidx + adjIndex;
                    // #1 calculate the mean
                    // iterating through the stack of images in the mini_batch
                    mean_accum += in_ptr[index];
                }
                mean_accum /= double(n_batchs);

                if(savemeanvar)
                    saveMean[adjIndex] = mean_accum;
                if(runningmeanvar)
                {
                    double newRunMean = runningMean[adjIndex] * (1 - expAvgFactor);
                    runningMean[adjIndex] =
                        mean_accum * expAvgFactor + newRunMean; // newMean*factor + tmp
                }

                elemStd        = 0.;
                variance_accum = 0.;
                // #2 calculate the variances
                // sigma^2 = (1/batch_mean) * sum( (x_i - batch_mean)^2 )
                for(int bidx = 0; bidx < n_batchs; bidx++)
                { // via mini_batch
                    // per (x-dims) channel load a block of data into LDS
                    index   = in_nstride * bidx + adjIndex;
                    elemStd = in_ptr[index] -
                              mean_accum; // (x_i - mean) //this is reused but needs recalc
                    variance_accum += elemStd * elemStd; // sum{ (x_i - mean)^2 }
                }                                        // end for(n)

                variance_accum /= double(n_batchs); // (1/N)*sum{ (x_i - mean)^2 }

                if(runningmeanvar)
                {
                    // var(n+1) = p * var(n-1) + (1 - p)*(b/b-1)*var(n)
                    double adjust =
                        (n_batchs == 1)
                            ? variance_accum
                            : (double(n_batchs) / double(n_batchs - 1) * variance_accum);
                    runningVariance[adjIndex] =
                        (1 - expAvgFactor) * runningVariance[cidx] + expAvgFactor * adjust;
                }

                // #3 add epsilon for numeric stability, sqr_root, and invert
                double elemInvVar = 1.0 / sqrt(variance_accum + epsilon);

                if(savemeanvar)
                    saveInvVariance[adjIndex] = elemInvVar; /*output only*/

                // #4 apply the normalization
                // x_hat = (x_i - mean) / sqrt(variance_accum - epsilon)
                for(int bidx = 0; bidx < n_batchs; bidx++)
                { // via mini_batch
                    index = in_nstride * bidx + adjIndex;
                    // per (x-dims) channel load a block of data into LDS
                    elemStd      = in_ptr[index] - mean_accum; // (x_i - mean)
                    double inhat = elemStd * elemInvVar;
                    // #5 Gamma and Beta adjust
                    // y_i = gamma*x_hat + beta
                    out_ptr[index] = scale_ptr[adjIndex] * inhat + bias_ptr[adjIndex];
                } // end for(n_batchs)
            }     // for (column)
        }         // for (row)
    }             // for (channel)
    return (ret);
}

template <typename T>
int miopenBNFwdTrainSpatialRunHost(
    /*    T alpha,
        T beta,
    */
    int n_batchs,
    int channels,
    int height,
    int width,
    const T* in_ptr,
    double* out_ptr,
    T* scale_ptr,
    T* bias_ptr,
    double epsilon,
    bool savemeanvar,
    bool runningmeanvar,
    double* saveMean,
    double* saveInvVariance,
    double* runningMean,
    double* runningVariance,
    double expAvgFactor)
{

    unsigned int imgIndex;
    unsigned int index;
    unsigned int adjIndex;
    unsigned int in_nstride = channels * height * width;
    unsigned int in_cstride = height * width;
    auto NHW                = double(in_cstride * n_batchs);

    double elemStd        = 0.;
    double variance_accum = 0.;
    double mean_accum     = 0.;

#if(MIO_HEIRARCH_SEL == 1)
    double variance_accum_arr[MIO_BN_DIST];
    double mean_accum_arr[MIO_BN_DIST];
#endif

    int ret = 0;
    for(int cidx = 0; cidx < channels; cidx++)
    { // via channel
        mean_accum = 0.;
#if(MIO_HEIRARCH_SEL == 1)
        for(int i = 0; i < MIO_BN_DIST; i++)
        {
            variance_accum_arr[i] = 0.;
            mean_accum_arr[i]     = 0.;
        }
#endif

#if(MIO_HEIRARCH_SEL == 0)
        // process the batch per channel
        for(int row = 0; row < height; row++)
        { // via rows
            for(int column = 0; column < width; column++)
            { // via columns
                imgIndex = width * row + column;
                adjIndex = in_cstride * cidx + imgIndex;
                if(imgIndex < in_cstride)
                {
                    for(int bidx = 0; bidx < n_batchs; bidx++)
                    { // via mini_batch
                        index = in_nstride * bidx + adjIndex;
                        // #1 calculate the mean
                        // iterating through the stack of images in the mini_batch
                        mean_accum += in_ptr[index];
                        // mean_accum += 1;
                    } // end for (n)
                }
            } // end for (column)
        }     // end for (row)
#else
        for(int im = 0; im < in_cstride; im += MIO_BN_DIST)
        {
            for(int i = 0; i < MIO_BN_DIST; i++)
            {
                imgIndex = im + i;
                adjIndex = in_cstride * cidx + imgIndex;
                if(imgIndex < in_cstride)
                {
                    for(int bidx = 0; bidx < n_batchs; bidx++)
                    { // via mini_batch
                        index = in_nstride * bidx + adjIndex;
                        // #1 calculate the mean
                        // iterating through the stack of images in the mini_batch
                        mean_accum_arr[i] += in_ptr[index];
                    } // end for (n)
                }
            }
        }
        for(int i = 0; i < MIO_BN_DIST; i++)
        {
            mean_accum += mean_accum_arr[i];
        }
#endif
        mean_accum /= NHW;

        if(savemeanvar)
            saveMean[cidx] = mean_accum;
        if(runningmeanvar)
        {
            double newRunMean = runningMean[cidx] * (1 - expAvgFactor);
            runningMean[cidx] = mean_accum * expAvgFactor + newRunMean; // newMean*factor + tmp
        }

        elemStd        = 0.;
        variance_accum = 0.;
#if(MIO_HEIRARCH_SEL == 0)
        // #2 calculate the variances
        // sigma^2 = (1/batch_mean) * sum( (x_i - batch_mean)^2 )
        for(int row = 0; row < height; row++)
        { // via rows
            for(int column = 0; column < width; column++)
            { // via columns
                imgIndex = width * row + column;
                adjIndex = in_cstride * cidx + imgIndex;
                if(imgIndex < in_cstride)
                {
                    for(int bidx = 0; bidx < n_batchs; bidx++)
                    { // via mini_batch
                        // per (x-dims) channel load a block of data into LDS
                        index = in_nstride * bidx + adjIndex;

                        // using out buffer as scratchpad
                        out_ptr[index] = elemStd =
                            (in_ptr[index] -
                             mean_accum); // (x_i - mean) //this is reused but needs recalc
                        variance_accum += (elemStd * elemStd); // sum{ (x_i - mean)^2 }
                    }                                          // end for(n)
                }
            } // end for (column)
        }     // end for (row)

#else
        for(int im = 0; im < in_cstride; im += MIO_BN_DIST)
        {
            for(int i = 0; i < MIO_BN_DIST; i++)
            {
                imgIndex = im + i;
                adjIndex = in_cstride * cidx + imgIndex;
                if(imgIndex < in_cstride)
                {
                    for(int bidx = 0; bidx < n_batchs; bidx++)
                    { // via mini_batch
                        // per (x-dims) channel load a block of data into LDS
                        index = in_nstride * bidx + adjIndex;

                        // using out buffer as scratchpad
                        elemStd = (in_ptr[index] -
                                   mean_accum); // (x_i - mean) //this is reused but needs recalc
                        variance_accum_arr[i] += (elemStd * elemStd); // sum{ (x_i - mean)^2 }
                    }
                }
            }
        } // end for (row)
        for(int i = 0; i < MIO_BN_DIST; i++)
        {
            variance_accum += variance_accum_arr[i];
        }
#endif

        variance_accum /= NHW; // (1/N)*sum{ (x_i - mean)^2 }
        // printf("Variance sum on host: %f\n",variance_accum);

        if(runningmeanvar)
        {
            double adjust = (n_batchs * in_cstride == 1) ? variance_accum
                                                         : (NHW / (NHW - 1.0) * variance_accum);
            runningVariance[cidx] =
                (1 - expAvgFactor) * runningVariance[cidx] + expAvgFactor * adjust;
        }

        // #3 add epsilon for numeric stability, sqr_root, and invert
        double invertVar = 1.0 / sqrt(variance_accum + epsilon);

        // printf("invVar on host: %lf\n",invertVar);

        if(savemeanvar)
            saveInvVariance[cidx] = invertVar; /*output only*/

        // #4 apply the normalization
        // x_hat = (x_i - mean) / sqrt(variance_accum + epsilon)
        for(int row = 0; row < height; row++)
        { // via rows
            for(int column = 0; column < width; column++)
            { // via columns
                imgIndex = width * row + column;
                adjIndex = in_cstride * cidx + imgIndex;
                if(imgIndex < in_cstride)
                {
                    for(int bidx = 0; bidx < n_batchs; bidx++)
                    { // via mini_batch
                        index = in_nstride * bidx + adjIndex;
                        // per (x-dims) channel load a block of data into LDS
                        // elemStd =(in_ptr[index] - mean_accum);
                        // double inhat = elemStd*invertVar;
                        // #5 Gamma and Beta adjust
                        // y_i = gamma*x_hat + beta
                        out_ptr[index] =
                            (scale_ptr[cidx] * (invertVar * out_ptr[index])) + bias_ptr[cidx];
                    } // end for(n_batchs)
                }
            } // for (column)
        }     // for (row)
    }         // for (channel)
    return (ret);
}

//====================== END TRAINING KERNELS =========================

//==================== BEGIN INFERENCE KERNELS ========================

template <typename T>
int miopenBNFwdInferPerActivationRunHost(
    /*	T alpha,
            T beta,
    */
    int n_batchs,
    int channels,
    int height,
    int width,
    const T* in_ptr,
    double* out_ptr,
    T* scale_ptr,
    T* bias_ptr,
    double epsilon,
    bool estmeanvar,
    double* estimatedMean,
    double* estimatedVariance)
{ // use running mean and variance

    // C*H*W is also stored as in_nstride, H*W is in_cstride, W is in_hstride.
    unsigned int index;
    unsigned int adjIndex;
    unsigned int in_nstride = channels * height * width;
    unsigned int in_cstride = height * width;

    double elemStd = 0.;

    int ret = 0;
    if(estmeanvar)
    {

        printf("Running estimated mean / var inference on CPU.\n");
        double mean     = 0.;
        double variance = 0.;
        for(int cidx = 0; cidx < channels; cidx++)
        { // via channel
            // process the batch per channel
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    adjIndex          = in_cstride * cidx + width * row + column;
                    mean              = estimatedMean[adjIndex];
                    variance          = estimatedVariance[adjIndex];
                    double elemInvVar = 1.0 / double(sqrt(variance + epsilon));
                    for(int bidx = 0; bidx < n_batchs; bidx++)
                    { // via mini_batch
                        index = in_nstride * bidx + adjIndex;
                        // per (x-dims) channel load a block of data into LDS
                        elemStd      = in_ptr[index] - mean; // (x_i - mean)
                        double inhat = elemStd * elemInvVar;
                        // #5 Gamma and Beta adjust
                        // y_i = gamma*x_hat + beta
                        out_ptr[index] = scale_ptr[adjIndex] * inhat + bias_ptr[adjIndex];
                    } // end for(n_batchs)
                }     // for (column)
            }
        }
    }
    else
    {

        double mean_accum     = 0.;
        double variance_accum = 0.;
        for(int cidx = 0; cidx < channels; cidx++)
        { // via channel
            // process the batch per channel
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    mean_accum = 0.;
                    adjIndex   = in_cstride * cidx + width * row + column;
                    for(int bidx = 0; bidx < n_batchs; bidx++)
                    { // via mini_batch
                        index = in_nstride * bidx + adjIndex;
                        // #1 calculate the mean
                        // iterating through the stack of images in the mini_batch
                        mean_accum += in_ptr[index];
                    }
                    mean_accum /= double(n_batchs);

                    elemStd        = 0.;
                    variance_accum = 0.;
                    // #2 calculate the variances
                    // sigma^2 = (1/batch_mean) * sum( (x_i - batch_mean)^2 )
                    for(int bidx = 0; bidx < n_batchs; bidx++)
                    { // via mini_batch
                        // per (x-dims) channel load a block of data into LDS
                        index   = in_nstride * bidx + adjIndex;
                        elemStd = in_ptr[index] - mean_accum; // (x_i - mean)
                        variance_accum += elemStd * elemStd;  // sum{ (x_i - mean)^2 }
                    }                                         // end for(n)
                    variance_accum /= double(n_batchs);       // (1/N)*sum{ (x_i - mean)^2 }

                    // #3 add epsilon for numeric stability, sqr_root, and invert
                    double elemInvVar = 1.0 / double(sqrt(variance_accum + epsilon));

                    // #4 apply the normalization
                    // x_hat = (x_i - mean) / sqrt(variance_accum - epsilon)
                    for(int bidx = 0; bidx < n_batchs; bidx++)
                    { // via mini_batch
                        index = in_nstride * bidx + adjIndex;
                        // per (x-dims) channel load a block of data into LDS
                        elemStd      = in_ptr[index] - mean_accum; // (x_i - mean)
                        double inhat = elemStd * elemInvVar;
                        // #5 Gamma and Beta adjust
                        // y_i = gamma*x_hat + beta
                        out_ptr[index] = scale_ptr[adjIndex] * inhat + bias_ptr[adjIndex];
                    } // end for(n_batchs)
                }     // for (column)
            }         // for (row)
        }             // for (channel)
    }
    return (ret);
}

template <typename T>
int miopenBNFwdInferSpatialRunHost(
    /*        T alpha,
            T beta,
    */
    int n_batchs,
    int channels,
    int height,
    int width,
    const T* in_ptr,
    double* out_ptr,
    T* scale_ptr,
    T* bias_ptr,
    double epsilon,
    bool estmeanvar,
    double* estimatedMean,
    double* estimatedVariance)
{

    unsigned int index;
    unsigned int adjIndex;
    unsigned int in_nstride = channels * height * width;
    unsigned int in_cstride = height * width;

    double elemStd = 0.;
    int ret        = 0;

    if(estmeanvar)
    {

        double variance = 0.;
        double mean     = 0.;
        double inhat    = 0.;
        for(int cidx = 0; cidx < channels; cidx++)
        { // via channel
            mean             = estimatedMean[cidx];
            variance         = estimatedVariance[cidx];
            double invertVar = 1.0 / sqrt(variance + epsilon);
            // process the batch per channel
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    adjIndex = in_cstride * cidx + width * row + column;
                    for(int bidx = 0; bidx < n_batchs; bidx++)
                    { // via mini_batch
                        index          = in_nstride * bidx + adjIndex;
                        elemStd        = in_ptr[index] - mean;
                        inhat          = elemStd * invertVar;
                        out_ptr[index] = scale_ptr[cidx] * inhat + bias_ptr[cidx];
                    } // end for (n)
                }
            }
        }
    }
    else
    {

#if(MIO_HEIRARCH_SEL == 1)
        double variance_accum_arr[MIO_BN_DIST];
        double mean_accum_arr[MIO_BN_DIST];
#endif

        double variance_accum = 0.;
        double mean_accum     = 0.;
        for(int cidx = 0; cidx < channels; cidx++)
        { // via channel
#if(MIO_HEIRARCH_SEL == 1)
            for(int i = 0; i < MIO_BN_DIST; i++)
            {
                variance_accum_arr[i] = 0.;
                mean_accum_arr[i]     = 0.;
            }
#endif

            mean_accum = 0.;
#if(MIO_HEIRARCH_SEL == 0)
            // process the batch per channel
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    adjIndex = in_cstride * cidx + width * row + column;
                    for(int bidx = 0; bidx < n_batchs; bidx++)
                    { // via mini_batch
                        index = in_nstride * bidx + adjIndex;
                        // #1 calculate the mean
                        // iterating through the stack of images in the mini_batch
                        mean_accum += in_ptr[index];
                    } // end for (n)
                }     // end for (column)
            }         // end for (row)
#else
            int imgIndex = 0;
            // process the batch per channel
            for(int im = 0; im < in_cstride; im += MIO_BN_DIST)
            {
                for(int i = 0; i < MIO_BN_DIST; i++)
                {
                    imgIndex = im + i;
                    adjIndex = in_cstride * cidx + imgIndex;
                    for(int bidx = 0; bidx < n_batchs; bidx++)
                    { // via mini_batch
                        index = in_nstride * bidx + adjIndex;
                        // #1 calculate the mean
                        // iterating through the stack of images in the mini_batch
                        mean_accum_arr[i] += in_ptr[index];
                    } // end for (n)
                }     // end for (column)
            }         // end for (row)
            for(int i = 0; i < MIO_BN_DIST; i++)
            {
                mean_accum += mean_accum_arr[i];
            }
#endif
            mean_accum /= double(in_cstride * n_batchs);

            elemStd        = 0.;
            variance_accum = 0.;
#if(MIO_HEIRARCH_SEL == 0)
            // #2 calculate the variances
            // sigma^2 = (1/batch_mean) * sum( (x_i - batch_mean)^2 )
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    adjIndex = in_cstride * cidx + width * row + column;
                    for(int bidx = 0; bidx < n_batchs; bidx++)
                    { // via mini_batch
                        // per (x-dims) channel load a block of data into LDS
                        index = in_nstride * bidx + adjIndex;

                        // using out buffer as scratchpad
                        out_ptr[index] = elemStd = (in_ptr[index] - mean_accum); // (x_i - mean)
                        variance_accum += (elemStd * elemStd); // sum{ (x_i - mean)^2 }
                    }                                          // end for(n)
                }                                              // end for (column)
            }                                                  // end for (row)
#else
            for(int im = 0; im < in_cstride; im += MIO_BN_DIST)
            {
                for(int i = 0; i < MIO_BN_DIST; i++)
                {
                    imgIndex = im + i;
                    adjIndex = in_cstride * cidx + imgIndex;
                    for(int bidx = 0; bidx < n_batchs; bidx++)
                    { // via mini_batch
                        // per (x-dims) channel load a block of data into LDS
                        index = in_nstride * bidx + adjIndex;

                        // using out buffer as scratchpad
                        out_ptr[index] = elemStd = (in_ptr[index] - mean_accum); // (x_i - mean)
                        variance_accum_arr[i] += (elemStd * elemStd); // sum{ (x_i - mean)^2 }
                    }                                                 // end for(n)
                }                                                     // end for
            }                                                         // end for
            for(int i = 0; i < MIO_BN_DIST; i++)
            {
                variance_accum += variance_accum_arr[i];
            }
#endif
            variance_accum /= double(in_cstride * n_batchs); // (1/N)*sum{ (x_i - mean)^2 }

            // #3 add epsilon for numeric stability, sqr_root, and invert
            double invertVar = 1.0 / sqrt(variance_accum + epsilon);

            // #4 apply the normalization
            // x_hat = (x_i - mean) / sqrt(variance_accum - epsilon)
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    adjIndex = in_cstride * cidx + width * row + column;
                    for(int bidx = 0; bidx < n_batchs; bidx++)
                    { // via mini_batch
                        index = in_nstride * bidx + adjIndex;
                        // per (x-dims) channel load a block of data into LDS
                        // elemStd = in_ptr[index] - mean_accum;// (x_i - mean)
                        elemStd      = out_ptr[index]; // using saved values from output tensor
                        double inhat = elemStd * invertVar;
                        // #5 Gamma and Beta adjust
                        // y_i = gamma*x_hat + beta
                        out_ptr[index] = scale_ptr[cidx] * inhat + bias_ptr[cidx];
                    } // end for(n_batchs)
                }     // for (column)
            }         // for (row)
        }             // for (channel)
    }                 // end if
    return (ret);
}

//================ END FWD INFERENCE ========================

//================ START BACKWARDS PASS =====================

template <typename T>
int miopenBNBwdPerActivationRunHost(
    /*        T alphaDiff,
            T betaDiff,
            T alphaParam,
            T betaParam,
    */
    int n_batchs,
    int channels,
    int height,
    int width,
    const T* x_ptr,  // layer's fwd input
    const T* dy_ptr, // fwd normalized x
    double* dx_ptr,
    T* scale_ptr,
    double* dscale_ptr,
    double* dbias_ptr,
    double epsilon,
    bool savedmeanvar,
    double* savedMean,
    double* savedInvVariance)
{

    // C*H*W is also stored as in_nstride, H*W is in_cstride, W is in_hstride.
    unsigned int index, xhat_index;
    unsigned int adjIndex;
    unsigned int in_nstride = channels * height * width;
    unsigned int in_cstride = height * width;
    double elemStd          = 0.;
    double mean             = 0.;
    double elemInvVar       = 0.;
    double dyelem           = 0.;
    double dxhat            = 0.;
    double dxhathat         = 0.;
    double tmp1, tmp2, tmp3;

    std::vector<double> xhat(n_batchs * in_cstride);

    if(savedmeanvar)
    {
        for(int cidx = 0; cidx < channels; cidx++)
        { // via channel
            // process the batch per channel
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns

                    adjIndex = in_cstride * cidx + width * row + column;

                    mean       = savedMean[adjIndex];        // HxW elements
                    elemInvVar = savedInvVariance[adjIndex]; // HxW elements

                    dxhat    = 0.;
                    dxhathat = 0.;
                    for(int bidx = 0; bidx < n_batchs; bidx++)
                    { // via mini_batch
                        index      = in_nstride * bidx + adjIndex;
                        xhat_index = in_cstride * bidx + (width * row + column);
                        // per (x-dims) channel load a block of data into LDS
                        elemStd          = x_ptr[index] - mean; // (x_i - mean)
                        xhat[xhat_index] = elemStd * elemInvVar;
                        dyelem           = dy_ptr[index];
                        dbias_ptr[adjIndex] += dyelem;
                        dscale_ptr[adjIndex] += xhat[xhat_index] * dyelem;
                        tmp1 = scale_ptr[adjIndex] * dyelem;
                        dxhat += tmp1;
                        dxhathat += tmp1 * xhat[xhat_index];
                    } // end for(n_batchs)

                    for(int bidx = 0; bidx < n_batchs; bidx++)
                    { // via mini_batch
                        index         = in_nstride * bidx + adjIndex;
                        xhat_index    = in_cstride * bidx + (width * row + column);
                        tmp1          = xhat[xhat_index] * dxhathat + dxhat;
                        tmp2          = n_batchs * dxhat - tmp1;
                        tmp3          = elemInvVar / (double(n_batchs));
                        dx_ptr[index] = tmp3 * tmp2;
                    } // end for(n_batchs)
                }     // for (column)
            }         // for (row)
        }             // for (cidx)
    }
    else
    {

        double variance = 0.;
        for(int cidx = 0; cidx < channels; cidx++)
        { // via channel
            // process the batch per channel
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    mean     = 0.;
                    adjIndex = in_cstride * cidx + width * row + column;
                    for(int bidx = 0; bidx < n_batchs; bidx++)
                    { // via mini_batch
                        index = in_nstride * bidx + adjIndex;
                        // #1 calculate the mean
                        // iterating through the stack of images in the mini_batch
                        mean += x_ptr[index];
                    }
                    mean /= double(n_batchs);

                    elemStd  = 0.;
                    variance = 0.;
                    // #2 calculate the variances
                    // sigma^2 = (1/batch_mean) * sum( (x_i - batch_mean)^2 )
                    for(int bidx = 0; bidx < n_batchs; bidx++)
                    { // via mini_batch
                        // per (x-dims) channel load a block of data into LDS
                        index   = in_nstride * bidx + adjIndex;
                        elemStd = x_ptr[index] - mean; // (x_i - mean)
                        variance += elemStd * elemStd; // sum{ (x_i - mean)^2 }
                    }                                  // end for(n)
                    variance /= double(n_batchs);      // (1/N)*sum{ (x_i - mean)^2 }

                    // #3 add epsilon for numeric stability, sqr_root, and invert
                    elemInvVar = 1.0 / sqrt(variance + epsilon);

                    dxhat    = 0.;
                    dxhathat = 0.;
                    for(int bidx = 0; bidx < n_batchs; bidx++)
                    { // via mini_batch
                        index      = in_nstride * bidx + adjIndex;
                        xhat_index = in_cstride * bidx + (width * row + column);
                        // per (x-dims) channel load a block of data into LDS
                        elemStd          = x_ptr[index] - mean; // (x_i - mean)
                        xhat[xhat_index] = elemStd * elemInvVar;
                        dyelem           = dy_ptr[index];
                        dbias_ptr[adjIndex] += dyelem;
                        dscale_ptr[adjIndex] += xhat[xhat_index] * dyelem;
                        tmp1 = scale_ptr[adjIndex] * dyelem;
                        dxhat += tmp1;
                        dxhathat += tmp1 * xhat[xhat_index];
                    } // end for(n_batchs)

                    for(int bidx = 0; bidx < n_batchs; bidx++)
                    { // via mini_batch
                        index         = in_nstride * bidx + adjIndex;
                        xhat_index    = in_cstride * bidx + (width * row + column);
                        tmp1          = xhat[xhat_index] * dxhathat + dxhat;
                        tmp2          = n_batchs * dxhat - tmp1;
                        tmp3          = elemInvVar / (double(n_batchs));
                        dx_ptr[index] = tmp3 * tmp2;
                    } // end for(n_batchs)
                }     // for (column)
            }         // for (row)
        }             // for (channel)
    }                 // end else

    return 0;
}

template <typename T>
int miopenBNBwdSpatialRunHost(
    /*      T alpha,
            T beta,
            T alphaParam,
            T betaParam,
    */
    int n_batchs,
    int channels,
    int height,
    int width,
    const T* x_ptr,  // layer's fwd input
    const T* dy_ptr, // fwd normalized x
    double* dx_ptr,
    T* scale_ptr,
    double* dscale_ptr,
    double* dbias_ptr,
    double epsilon,
    bool savedmeanvar,
    double* savedMean,
    double* savedInvVariance)
{

    // C*H*W is also stored as in_nstride, H*W is in_cstride, W is in_hstride.
    unsigned int index;
    unsigned int adjIndex;
    unsigned int in_nstride = channels * height * width;
    unsigned int in_cstride = height * width;
    unsigned int Csubindex  = 0;
    double elemStd          = 0.;
    double mean             = 0.;
    double invVar           = 0.;
    double dyelem           = 0.;
    double NHW              = double(n_batchs * in_cstride);

    if(savedmeanvar)
    {
        for(int cidx = 0; cidx < channels; cidx++)
        { // via channel
            Csubindex = in_cstride * cidx;
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    adjIndex = Csubindex + width * row + column;

                    mean   = savedMean[cidx];        // 1xCx1x1 elements
                    invVar = savedInvVariance[cidx]; // 1xCx1x1 elements

                    for(int bidx = 0; bidx < n_batchs; bidx++)
                    { // via mini_batch
                        index = in_nstride * bidx + adjIndex;
                        // per (x-dims) channel load a block of data into LDS
                        elemStd = x_ptr[index] - mean; // (x_i - mean)
                        dyelem  = dy_ptr[index];
                        dbias_ptr[cidx] += dyelem;
                        dscale_ptr[cidx] += elemStd * invVar * dyelem;
                    } // end for(n_batchs)
                }     // for (column)
            }         // for (row)

            // process the batch per channel
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    adjIndex = Csubindex + width * row + column;

                    for(int bidx = 0; bidx < n_batchs; bidx++)
                    { // via mini_batch

                        index         = in_nstride * bidx + adjIndex;
                        elemStd       = x_ptr[index] - mean; // (x_i - mean)
                        double tmp1   = NHW * dy_ptr[index] - dbias_ptr[cidx];
                        double tmp2   = -elemStd * invVar * dscale_ptr[cidx];
                        double tmp3   = (scale_ptr[cidx] * invVar) / NHW;
                        dx_ptr[index] = tmp3 * (tmp2 + tmp1);
                    } // end for(n_batchs)
                }     // for (column)
            }         // for (row)
        }             // for (cidx)
    }
    else
    {

#if(MIO_HEIRARCH_SEL == 1)
        double variance_accum_arr[MIO_BN_DIST];
        double mean_accum_arr[MIO_BN_DIST];
        double dbias_accum_arr[MIO_BN_DIST];
        double dscale_accum_arr[MIO_BN_DIST];
#else
        std::vector<double> xhat(n_batchs * in_cstride);
        unsigned int xhat_index;
#endif

        double variance = 0.;
        for(int cidx = 0; cidx < channels; cidx++)
        { // via channel
#if(MIO_HEIRARCH_SEL == 1)
            for(int i = 0; i < MIO_BN_DIST; i++)
            {
                variance_accum_arr[i] = 0.;
                mean_accum_arr[i]     = 0.;
                dbias_accum_arr[i]    = 0.;
                dscale_accum_arr[i]   = 0.;
            }
#endif
            Csubindex = in_cstride * cidx;

            // process the batch per channel
            mean = 0.;
#if(MIO_HEIRARCH_SEL == 0)
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    adjIndex = Csubindex + width * row + column;

                    for(int bidx = 0; bidx < n_batchs; bidx++)
                    { // via mini_batch
                        index = in_nstride * bidx + adjIndex;
                        // #1 calculate the mean
                        // iterating through the stack of images in the mini_batch
                        mean += x_ptr[index];
                    }
                } // for (column)
            }     // for (row)
#else
            int imgIndex = 0;
            for(int im = 0; im < in_cstride; im += MIO_BN_DIST)
            {
                for(int i = 0; i < MIO_BN_DIST; i++)
                {
                    imgIndex = im + i;
                    adjIndex = in_cstride * cidx + imgIndex;
                    if(imgIndex < in_cstride)
                    {
                        for(int bidx = 0; bidx < n_batchs; bidx++)
                        { // via mini_batch
                            index = in_nstride * bidx + adjIndex;
                            // #1 calculate the mean
                            // iterating through the stack of images in the mini_batch
                            mean_accum_arr[i] += x_ptr[index];
                        }
                    }
                } // end for
            }     // end for
            for(int i = 0; i < MIO_BN_DIST; i++)
            {
                mean += mean_accum_arr[i];
            }
#endif
            mean /= NHW;
            // printf("MEAN: %f\n",mean);
            elemStd  = 0.;
            variance = 0.;
#if(MIO_HEIRARCH_SEL == 0)
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    adjIndex = Csubindex + width * row + column;

                    // #2 calculate the variances
                    // sigma^2 = (1/batch_mean) * sum( (x_i - batch_mean)^2 )
                    for(int bidx = 0; bidx < n_batchs; bidx++)
                    { // via mini_batch
                        // per (x-dims) channel load a block of data into LDS
                        index   = in_nstride * bidx + adjIndex;
                        elemStd = x_ptr[index] - mean; // (x_i - mean)
                        variance += elemStd * elemStd; // sum{ (x_i - mean)^2 }
                    }                                  // end for(n)
                }                                      // for (column)
            }                                          // for (row)
#else
            for(int im = 0; im < in_cstride; im += MIO_BN_DIST)
            {
                for(int i = 0; i < MIO_BN_DIST; i++)
                {
                    imgIndex = im + i;
                    adjIndex = in_cstride * cidx + imgIndex;
                    // #2 calculate the variances
                    // sigma^2 = (1/batch_mean) * sum( (x_i - batch_mean)^2 )
                    for(int bidx = 0; bidx < n_batchs; bidx++)
                    { // via mini_batch
                        // per (x-dims) channel load a block of data into LDS
                        index = in_nstride * bidx + adjIndex;
                        if(imgIndex < in_cstride)
                        {
                            elemStd = x_ptr[index] - mean;              // (x_i - mean)
                            variance_accum_arr[i] += elemStd * elemStd; // sum{ (x_i - mean)^2 }
                        }
                    } // end for
                }     // end for
            }
            for(int i = 0; i < MIO_BN_DIST; i++)
            {
                variance += variance_accum_arr[i];
            }
#endif
            variance /= NHW; // (1/(N*H*W))*sum{ (x_i - mean)^2 }
            // printf("VARIANCE: %f\n",variance);
            // #3 add epsilon for numeric stability, sqr_root, and invert
            invVar = 1. / sqrt(variance + epsilon);
            // printf("invVar: %f\n",invVar);

            dscale_ptr[cidx] = 0.;
            dbias_ptr[cidx]  = 0.;
#if(MIO_HEIRARCH_SEL == 0)
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    adjIndex = Csubindex + width * row + column;

                    for(int bidx = 0; bidx < n_batchs; bidx++)
                    { // via mini_batch
                        index      = in_nstride * bidx + adjIndex;
                        xhat_index = in_cstride * bidx + (width * row + column);
                        // per (x-dims) channel load a block of data into LDS
                        elemStd          = x_ptr[index] - mean; // (x_i - mean)
                        xhat[xhat_index] = elemStd * invVar;
                        dyelem           = dy_ptr[index];
                        dbias_ptr[cidx] += dyelem;
                        dscale_ptr[cidx] += xhat[xhat_index] * dyelem;
                    } // end for(n_batchs)
                }     // for (column)
            }         // for (row)
#else
            for(int im = 0; im < in_cstride; im += MIO_BN_DIST)
            {
                for(int i = 0; i < MIO_BN_DIST; i++)
                {
                    imgIndex = im + i;
                    adjIndex = in_cstride * cidx + imgIndex;
                    for(int bidx = 0; bidx < n_batchs; bidx++)
                    { // via mini_batch
                        index = in_nstride * bidx + adjIndex;
                        // per (x-dims) channel load a block of data into LDS
                        if(imgIndex < in_cstride)
                        {
                            elemStd = x_ptr[index] - mean; // (x_i - mean)
                            dyelem  = dy_ptr[index];
                            dbias_accum_arr[i] += dyelem;
                            dscale_accum_arr[i] += elemStd * invVar * dyelem;
                        }
                    } // end for(n_batchs)
                }     // for (column)
            }         // for (row)
            for(int i = 0; i < MIO_BN_DIST; i++)
            {
                dbias_ptr[cidx] += dbias_accum_arr[i];
                dscale_ptr[cidx] += dscale_accum_arr[i];
            }
#endif

// printf("dscale: %f\n",dscale_ptr[cidx]);
// printf("dbias: %f\n",dbias_ptr[cidx]);

#if(MIO_HEIRARCH_SEL == 0)
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    adjIndex = Csubindex + width * row + column;
                    for(int bidx = 0; bidx < n_batchs; bidx++)
                    { // via mini_batch
                        index         = in_nstride * bidx + adjIndex;
                        xhat_index    = in_cstride * bidx + (width * row + column);
                        double tmp1   = NHW * dy_ptr[index] - dbias_ptr[cidx];
                        double tmp2   = -xhat[xhat_index] * dscale_ptr[cidx];
                        double tmp3   = (scale_ptr[cidx] * invVar) / NHW;
                        dx_ptr[index] = tmp3 * (tmp2 + tmp1);
                    } // end for(n_batchs)
                }     // for (column)
            }         // for (row)
#else
            for(int im = 0; im < in_cstride; im += MIO_BN_DIST)
            {
                for(int i = 0; i < MIO_BN_DIST; i++)
                {
                    imgIndex = im + i;
                    adjIndex = in_cstride * cidx + imgIndex;
                    for(int bidx = 0; bidx < n_batchs; bidx++)
                    { // via mini_batch
                        index = in_nstride * bidx + adjIndex;
                        if(imgIndex < in_cstride)
                        {
                            elemStd       = x_ptr[index] - mean; // (x_i - mean)
                            double tmp1   = NHW * dy_ptr[index] - dbias_ptr[cidx];
                            double tmp2   = -elemStd * invVar * dscale_ptr[cidx];
                            double tmp3   = (scale_ptr[cidx] * invVar) / NHW;
                            dx_ptr[index] = tmp3 * (tmp2 + tmp1);
                        }
                    } // end for(n_batchs)
                }     // for (column)
            }         // for (row)
#endif

        } // for (channel)
    }     // end else

    return 0;
}

#endif
