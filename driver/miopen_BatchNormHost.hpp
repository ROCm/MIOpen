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

template <typename Tgpu, typename Tref>
int miopenBNFwdTrainPerActivationRunHost(
    /*
        T alpha,
        T beta,
    */
    int n_batchs,
    int channels,
    int depth,
    int height,
    int width,
    const Tgpu* in_ptr,
    Tref* out_ptr,
    Tref* scale_ptr,
    Tref* bias_ptr,
    Tref epsilon,
    bool savemeanvar,
    bool runningmeanvar,
    Tref* saveMean,
    Tref* saveInvVariance,
    Tref* runningMean,
    Tref* runningVariance,
    Tref expAvgFactor)
{

    // C*H*W is also stored as in_nstride, H*W is in_cstride, W is in_hstride.
    unsigned int index;
    unsigned int adjIndex;
    unsigned int in_dstride = height * width;
    unsigned int in_cstride = depth * in_dstride;
    unsigned int in_nstride = channels * in_cstride;

    Tref mean_accum     = static_cast<Tref>(0.);
    Tref variance_accum = static_cast<Tref>(0.);
    Tref elemStd        = static_cast<Tref>(0.);

    int ret = 0;
    for(int cidx = 0; cidx < channels; cidx++)
    { // via channel
        mean_accum     = static_cast<Tref>(0.);
        variance_accum = static_cast<Tref>(0.);
        // process the batch per channel
        for(int didx = 0; didx < depth; didx++)
        { // via depth
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    mean_accum = static_cast<Tref>(0.);
                    adjIndex   = in_cstride * cidx + in_dstride * didx + width * row + column;
                    for(int bidx = 0; bidx < n_batchs; bidx++)
                    { // via mini_batch
                        index = in_nstride * bidx + adjIndex;
                        // #1 calculate the mean
                        // iterating through the stack of images in the mini_batch
                        mean_accum += in_ptr[index];
                    }
                    mean_accum /= static_cast<Tref>(n_batchs);

                    if(savemeanvar)
                        saveMean[adjIndex] = mean_accum;
                    if(runningmeanvar)
                    {
                        Tref newRunMean =
                            runningMean[adjIndex] * (static_cast<Tref>(1) - expAvgFactor);
                        runningMean[adjIndex] =
                            mean_accum * expAvgFactor + newRunMean; // newMean*factor + tmp
                    }

                    elemStd        = static_cast<Tref>(0.);
                    variance_accum = static_cast<Tref>(0.);
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

                    variance_accum /= static_cast<Tref>(n_batchs); // (1/N)*sum{ (x_i - mean)^2 }

                    if(runningmeanvar)
                    {
                        // var(n+1) = p * var(n-1) + (1 - p)*(b/b-1)*var(n)
                        Tref adjust = (n_batchs == 1)
                                          ? variance_accum
                                          : (static_cast<Tref>(n_batchs) /
                                             static_cast<Tref>(n_batchs - 1) * variance_accum);
                        runningVariance[adjIndex] =
                            (static_cast<Tref>(1) - expAvgFactor) * runningVariance[cidx] +
                            expAvgFactor * adjust;
                    }

                    // #3 add epsilon for numeric stability, sqr_root, and invert
                    Tref elemInvVar = static_cast<Tref>(1.0) / sqrt(variance_accum + epsilon);

                    if(savemeanvar)
                        saveInvVariance[adjIndex] = elemInvVar; /*output only*/

                    // #4 apply the normalization
                    // x_hat = (x_i - mean) / sqrt(variance_accum - epsilon)
                    for(int bidx = 0; bidx < n_batchs; bidx++)
                    { // via mini_batch
                        index = in_nstride * bidx + adjIndex;
                        // per (x-dims) channel load a block of data into LDS
                        elemStd    = in_ptr[index] - mean_accum; // (x_i - mean)
                        Tref inhat = elemStd * elemInvVar;
                        // #5 Gamma and Beta adjust
                        // y_i = gamma*x_hat + beta
                        out_ptr[index] = scale_ptr[adjIndex] * inhat + bias_ptr[adjIndex];
                    } // end for(n_batchs)
                }     // for (column)
            }         // for (row)
        }             // for (depth)
    }                 // for (channel)
    return (ret);
}

template <typename Tgpu, typename Tref>
int miopenBNFwdTrainSpatialRunHost(
    /*    T alpha,
        T beta,
    */
    int n_batchs,
    int channels,
    int depth,
    int height,
    int width,
    const Tgpu* in_ptr,
    Tref* out_ptr,
    Tref* scale_ptr,
    Tref* bias_ptr,
    Tref epsilon,
    bool savemeanvar,
    bool runningmeanvar,
    Tref* saveMean,
    Tref* saveInvVariance,
    Tref* runningMean,
    Tref* runningVariance,
    Tref expAvgFactor)
{

    unsigned int imgIndex;
    unsigned int index;
    unsigned int adjIndex;
    unsigned int in_dstride = height * width;
    unsigned int in_cstride = depth * in_dstride;
    unsigned int in_nstride = channels * in_cstride;
    auto NHW                = static_cast<Tref>(in_cstride * n_batchs);

    Tref elemStd        = static_cast<Tref>(0.);
    Tref variance_accum = static_cast<Tref>(0.);
    Tref mean_accum     = static_cast<Tref>(0.);

#if(MIO_HEIRARCH_SEL == 1)
    Tref variance_accum_arr[MIO_BN_DIST];
    Tref mean_accum_arr[MIO_BN_DIST];
#endif

    int ret = 0;
    for(int cidx = 0; cidx < channels; cidx++)
    { // via channel
        mean_accum = static_cast<Tref>(0.);
#if(MIO_HEIRARCH_SEL == 1)
        for(int i = 0; i < MIO_BN_DIST; i++)
        {
            variance_accum_arr[i] = static_cast<Tref>(0.);
            mean_accum_arr[i]     = static_cast<Tref>(0.);
        }
#endif

#if(MIO_HEIRARCH_SEL == 0)
        // process the batch per channel
        for(int didx = 0; didx < depth; didx++)
        { // depth
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    imgIndex = width * row + column;
                    adjIndex = in_cstride * cidx + in_dstride * didx + imgIndex;
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
        }         // end for (depth)
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
        mean_accum /= static_cast<Tref>(NHW);

        if(savemeanvar)
            saveMean[cidx] = mean_accum;
        if(runningmeanvar)
        {
            Tref newRunMean   = runningMean[cidx] * (static_cast<Tref>(1) - expAvgFactor);
            runningMean[cidx] = mean_accum * expAvgFactor + newRunMean; // newMean*factor + tmp
        }

        elemStd        = static_cast<Tref>(0.);
        variance_accum = static_cast<Tref>(0.);
#if(MIO_HEIRARCH_SEL == 0)
        // #2 calculate the variances
        // sigma^2 = (1/batch_mean) * sum( (x_i - batch_mean)^2 )
        for(int didx = 0; didx < depth; didx++)
        { // depth
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
        }         // end for (depth)

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

        variance_accum /= static_cast<Tref>(NHW); // (1/N)*sum{ (x_i - mean)^2 }
        // printf("Variance sum on host: %f\n",variance_accum);

        if(runningmeanvar)
        {
            Tref adjust =
                (n_batchs * in_cstride == 1)
                    ? variance_accum
                    : (static_cast<Tref>(NHW) / (static_cast<Tref>(NHW - 1.0)) * variance_accum);
            runningVariance[cidx] = (static_cast<Tref>(1) - expAvgFactor) * runningVariance[cidx] +
                                    expAvgFactor * adjust;
        }

        // #3 add epsilon for numeric stability, sqr_root, and invert
        Tref invertVar = static_cast<Tref>(1.0) / sqrt(variance_accum + epsilon);

        // printf("invVar on host: %lf\n",invertVar);

        if(savemeanvar)
            saveInvVariance[cidx] = invertVar; /*output only*/

        // #4 apply the normalization
        // x_hat = (x_i - mean) / sqrt(variance_accum + epsilon)
        for(int didx = 0; didx < depth; didx++)
        { // depth

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
                            // Tref inhat = elemStd*invertVar;
                            // #5 Gamma and Beta adjust
                            // y_i = gamma*x_hat + beta
                            out_ptr[index] =
                                (scale_ptr[cidx] * (invertVar * out_ptr[index])) + bias_ptr[cidx];
                        } // end for(n_batchs)
                    }
                } // for (column)
            }     // for (row)
        }         // for (depth)
    }             // for (channel)
    return (ret);
}

//====================== END TRAINING KERNELS =========================

//==================== BEGIN INFERENCE KERNELS ========================

template <typename Tgpu, typename Tref>
int miopenBNFwdInferPerActivationRunHost(
    /* T alpha,
            T beta,
    */
    int n_batchs,
    int channels,
    int depth,
    int height,
    int width,
    const Tgpu* in_ptr,
    Tref* out_ptr,
    Tref* scale_ptr,
    Tref* bias_ptr,
    Tref epsilon,
    bool estmeanvar,
    Tref* estimatedMean,
    Tref* estimatedVariance)
{ // use running mean and variance

    // C*H*W is also stored as in_nstride, H*W is in_cstride, W is in_hstride.
    unsigned int index;
    unsigned int adjIndex;
    unsigned int in_dstride = height * width;
    unsigned int in_cstride = depth * in_dstride;
    unsigned int in_nstride = channels * in_cstride;

    Tref elemStd = static_cast<Tref>(0.);

    int ret = 0;
    if(estmeanvar)
    {

        printf("Running estimated mean / var inference on CPU.\n");
        Tref mean     = static_cast<Tref>(0.);
        Tref variance = static_cast<Tref>(0.);
        for(int cidx = 0; cidx < channels; cidx++)
        { // via channel
            // process the batch per channel
            for(int didx = 0; didx < depth; didx++)
            { // via depth
                for(int row = 0; row < height; row++)
                { // via rows
                    for(int column = 0; column < width; column++)
                    { // via columns
                        adjIndex = in_cstride * cidx + in_dstride * didx + width * row + column;
                        mean     = estimatedMean[adjIndex];
                        variance = estimatedVariance[adjIndex];
                        Tref elemInvVar =
                            static_cast<Tref>(1.0) / static_cast<Tref>(sqrt(variance + epsilon));
                        for(int bidx = 0; bidx < n_batchs; bidx++)
                        { // via mini_batch
                            index = in_nstride * bidx + adjIndex;
                            // per (x-dims) channel load a block of data into LDS
                            elemStd    = in_ptr[index] - mean; // (x_i - mean)
                            Tref inhat = elemStd * elemInvVar;
                            // #5 Gamma and Beta adjust
                            // y_i = gamma*x_hat + beta
                            out_ptr[index] = scale_ptr[adjIndex] * inhat + bias_ptr[adjIndex];
                        } // end for(n_batchs)
                    }     // for (column)
                }
            }
        }
    }
    else
    {

        Tref mean_accum     = static_cast<Tref>(0.);
        Tref variance_accum = static_cast<Tref>(0.);
        for(int cidx = 0; cidx < channels; cidx++)
        { // via channel
            // process the batch per channel
            for(int didx = 0; didx < depth; didx++)
            { // via depth
                for(int row = 0; row < height; row++)
                { // via rows
                    for(int column = 0; column < width; column++)
                    { // via columns
                        mean_accum = static_cast<Tref>(0.);
                        adjIndex   = in_cstride * cidx + in_dstride * didx + width * row + column;
                        for(int bidx = 0; bidx < n_batchs; bidx++)
                        { // via mini_batch
                            index = in_nstride * bidx + adjIndex;
                            // #1 calculate the mean
                            // iterating through the stack of images in the mini_batch
                            mean_accum += in_ptr[index];
                        }
                        mean_accum /= static_cast<Tref>(n_batchs);

                        elemStd        = static_cast<Tref>(0.);
                        variance_accum = static_cast<Tref>(0.);
                        // #2 calculate the variances
                        // sigma^2 = (1/batch_mean) * sum( (x_i - batch_mean)^2 )
                        for(int bidx = 0; bidx < n_batchs; bidx++)
                        { // via mini_batch
                            // per (x-dims) channel load a block of data into LDS
                            index   = in_nstride * bidx + adjIndex;
                            elemStd = in_ptr[index] - mean_accum; // (x_i - mean)
                            variance_accum += elemStd * elemStd;  // sum{ (x_i - mean)^2 }
                        }                                         // end for(n)
                        variance_accum /=
                            static_cast<Tref>(n_batchs); // (1/N)*sum{ (x_i - mean)^2 }

                        // #3 add epsilon for numeric stability, sqr_root, and invert
                        Tref elemInvVar = static_cast<Tref>(1.0) /
                                          static_cast<Tref>(sqrt(variance_accum + epsilon));

                        // #4 apply the normalization
                        // x_hat = (x_i - mean) / sqrt(variance_accum - epsilon)
                        for(int bidx = 0; bidx < n_batchs; bidx++)
                        { // via mini_batch
                            index = in_nstride * bidx + adjIndex;
                            // per (x-dims) channel load a block of data into LDS
                            elemStd    = in_ptr[index] - mean_accum; // (x_i - mean)
                            Tref inhat = elemStd * elemInvVar;
                            // #5 Gamma and Beta adjust
                            // y_i = gamma*x_hat + beta
                            out_ptr[index] = scale_ptr[adjIndex] * inhat + bias_ptr[adjIndex];
                        } // end for(n_batchs)
                    }     // for (column)
                }         // for (row)
            }
        } // for (channel)
    }
    return (ret);
}

template <typename Tgpu, typename Tref>
int miopenBNFwdInferSpatialRunHost(
    /*        T alpha,
            T beta,
    */
    int n_batchs,
    int channels,
    int depth,
    int height,
    int width,
    const Tgpu* in_ptr,
    Tref* out_ptr,
    Tref* scale_ptr,
    Tref* bias_ptr,
    Tref epsilon,
    bool estmeanvar,
    Tref* estimatedMean,
    Tref* estimatedVariance)
{

    unsigned int index;
    unsigned int adjIndex;
    unsigned int in_dstride = height * width;
    unsigned int in_cstride = depth * in_dstride;
    unsigned int in_nstride = channels * in_cstride;

    Tref elemStd = static_cast<Tref>(0.);
    int ret      = 0;

    if(estmeanvar)
    {

        Tref variance = static_cast<Tref>(0.);
        Tref mean     = static_cast<Tref>(0.);
        Tref inhat    = static_cast<Tref>(0.);
        for(int cidx = 0; cidx < channels; cidx++)
        { // via channel
            mean           = estimatedMean[cidx];
            variance       = estimatedVariance[cidx];
            Tref invertVar = static_cast<Tref>(1.0) / static_cast<Tref>(sqrt(variance + epsilon));
            // process the batch per channel
            for(int didx = 0; didx < depth; didx++)
            { // via depth
                for(int row = 0; row < height; row++)
                { // via rows
                    for(int column = 0; column < width; column++)
                    { // via columns
                        adjIndex = in_cstride * cidx + in_dstride * didx + width * row + column;
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
    }
    else
    {

#if(MIO_HEIRARCH_SEL == 1)
        Tref variance_accum_arr[MIO_BN_DIST];
        Tref mean_accum_arr[MIO_BN_DIST];
#endif

        Tref variance_accum = static_cast<Tref>(0.);
        Tref mean_accum     = static_cast<Tref>(0.);
        for(int cidx = 0; cidx < channels; cidx++)
        { // via channel
#if(MIO_HEIRARCH_SEL == 1)
            for(int i = 0; i < MIO_BN_DIST; i++)
            {
                variance_accum_arr[i] = static_cast<Tref>(0.);
                mean_accum_arr[i]     = static_cast<Tref>(0.);
            }
#endif

            mean_accum = static_cast<Tref>(0.);
#if(MIO_HEIRARCH_SEL == 0)
            // process the batch per channel
            for(int didx = 0; didx < depth; didx++)
            { // via depth
                for(int row = 0; row < height; row++)
                { // via rows
                    for(int column = 0; column < width; column++)
                    { // via columns
                        adjIndex = in_cstride * cidx + in_dstride * didx + width * row + column;
                        for(int bidx = 0; bidx < n_batchs; bidx++)
                        { // via mini_batch
                            index = in_nstride * bidx + adjIndex;
                            // #1 calculate the mean
                            // iterating through the stack of images in the mini_batch
                            mean_accum += in_ptr[index];
                        } // end for (n)
                    }     // end for (column)
                }         // end for (row)
            }             // end for (depth)
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
            mean_accum /= static_cast<Tref>(in_cstride * n_batchs);

            elemStd        = static_cast<Tref>(0.);
            variance_accum = static_cast<Tref>(0.);
#if(MIO_HEIRARCH_SEL == 0)
            // #2 calculate the variances
            // sigma^2 = (1/batch_mean) * sum( (x_i - batch_mean)^2 )
            for(int didx = 0; didx < depth; didx++)
            { // via depth
                for(int row = 0; row < height; row++)
                { // via rows
                    for(int column = 0; column < width; column++)
                    { // via columns
                        adjIndex = in_cstride * cidx + in_dstride * didx + width * row + column;
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
            }
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
            variance_accum /=
                static_cast<Tref>(in_cstride * n_batchs); // (1/N)*sum{ (x_i - mean)^2 }

            // #3 add epsilon for numeric stability, sqr_root, and invert
            Tref invertVar =
                static_cast<Tref>(1.0) / static_cast<Tref>(sqrt(variance_accum + epsilon));

            // #4 apply the normalization
            // x_hat = (x_i - mean) / sqrt(variance_accum - epsilon)
            for(int didx = 0; didx < depth; didx++)
            { // via depth
                for(int row = 0; row < height; row++)
                { // via rows
                    for(int column = 0; column < width; column++)
                    { // via columns
                        adjIndex = in_cstride * cidx + in_dstride * didx + width * row + column;
                        for(int bidx = 0; bidx < n_batchs; bidx++)
                        { // via mini_batch
                            index = in_nstride * bidx + adjIndex;
                            // per (x-dims) channel load a block of data into LDS
                            // elemStd = in_ptr[index] - mean_accum;// (x_i - mean)
                            elemStd    = out_ptr[index]; // using saved values from output tensor
                            Tref inhat = elemStd * invertVar;
                            // #5 Gamma and Beta adjust
                            // y_i = gamma*x_hat + beta
                            out_ptr[index] = scale_ptr[cidx] * inhat + bias_ptr[cidx];
                        } // end for(n_batchs)
                    }     // for (column)
                }         // for (row)
            }             // for
        }                 // for (channel)
    }                     // end if
    return (ret);
}

//================ END FWD INFERENCE ========================

//================ START BACKWARDS PASS =====================

template <typename Tgpu, typename Tref, typename Tmix>
int miopenBNBwdPerActivationRunHost(
    /*        T alphaDiff,
            T betaDiff,
            T alphaParam,
            T betaParam,
    */
    int n_batchs,
    int channels,
    int depth,
    int height,
    int width,
    const Tgpu* x_ptr,  // layer's fwd input
    const Tgpu* dy_ptr, // fwd normalized x
    Tref* dx_ptr,
    Tmix* scale_ptr,
    Tref* dscale_ptr,
    Tref* dbias_ptr,
    Tref epsilon,
    bool savedmeanvar,
    Tref* savedMean,
    Tref* savedInvVariance)
{

    // C*H*W is also stored as in_nstride, H*W is in_cstride, W is in_hstride.
    unsigned int index, xhat_index;
    unsigned int adjIndex;
    unsigned int in_dstride = height * width;
    unsigned int in_cstride = depth * in_dstride;
    unsigned int in_nstride = channels * in_cstride;
    Tref elemStd            = static_cast<Tref>(0.);
    Tref mean               = static_cast<Tref>(0.);
    Tref elemInvVar         = static_cast<Tref>(0.);
    Tref dyelem             = static_cast<Tref>(0.);
    Tref dxhat              = static_cast<Tref>(0.);
    Tref dxhathat           = static_cast<Tref>(0.);
    Tref tmp1, tmp2, tmp3;

    std::vector<Tref> xhat(n_batchs * in_cstride);
    // When depth is present, flatten depth and height as height
    if(depth)
        height *= depth;

    if(savedmeanvar)
    {
        for(int cidx = 0; cidx < channels; cidx++)
        { // via channel
            for(int didx = 0; didx < depth; didx++)
            { // via depth
                // process the batch per channel
                for(int row = 0; row < height; row++)
                { // via rows
                    for(int column = 0; column < width; column++)
                    { // via columns

                        adjIndex = in_cstride * cidx + in_dstride * didx + width * row + column;

                        mean       = savedMean[adjIndex];        // HxW elements
                        elemInvVar = savedInvVariance[adjIndex]; // HxW elements

                        dxhat    = static_cast<Tref>(0.);
                        dxhathat = static_cast<Tref>(0.);
                        for(int bidx = 0; bidx < n_batchs; bidx++)
                        { // via mini_batch
                            index = in_nstride * bidx + adjIndex;
                            xhat_index =
                                in_cstride * bidx + in_dstride * didx + width * row + column;
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
                            index = in_nstride * bidx + adjIndex;
                            xhat_index =
                                in_cstride * bidx + in_dstride * didx + width * row + column;
                            tmp1          = xhat[xhat_index] * dxhathat + dxhat;
                            tmp2          = n_batchs * dxhat - tmp1;
                            tmp3          = elemInvVar / static_cast<Tref>(n_batchs);
                            dx_ptr[index] = tmp3 * tmp2;
                        } // end for(n_batchs)
                    }     // for (column)
                }         // for (row)
            }             // for (didx)
        }                 // for (cidx)
    }
    else
    {

        Tref variance = static_cast<Tref>(0.);
        for(int cidx = 0; cidx < channels; cidx++)
        { // via channel
            for(int didx = 0; didx < depth; didx++)
            { // via depth
                // process the batch per channel
                for(int row = 0; row < height; row++)
                { // via rows
                    for(int column = 0; column < width; column++)
                    { // via columns
                        mean     = static_cast<Tref>(0.);
                        adjIndex = in_cstride * cidx + in_dstride * didx + width * row + column;
                        for(int bidx = 0; bidx < n_batchs; bidx++)
                        { // via mini_batch
                            index = in_nstride * bidx + adjIndex;
                            // #1 calculate the mean
                            // iterating through the stack of images in the mini_batch
                            mean += x_ptr[index];
                        }
                        mean /= static_cast<Tref>(n_batchs);

                        elemStd  = static_cast<Tref>(0.);
                        variance = static_cast<Tref>(0.);
                        // #2 calculate the variances
                        // sigma^2 = (1/batch_mean) * sum( (x_i - batch_mean)^2 )
                        for(int bidx = 0; bidx < n_batchs; bidx++)
                        { // via mini_batch
                            // per (x-dims) channel load a block of data into LDS
                            index   = in_nstride * bidx + adjIndex;
                            elemStd = x_ptr[index] - mean;       // (x_i - mean)
                            variance += elemStd * elemStd;       // sum{ (x_i - mean)^2 }
                        }                                        // end for(n)
                        variance /= static_cast<Tref>(n_batchs); // (1/N)*sum{ (x_i - mean)^2 }

                        // #3 add epsilon for numeric stability, sqr_root, and invert
                        elemInvVar =
                            static_cast<Tref>(1.0) / static_cast<Tref>(sqrt(variance + epsilon));

                        dxhat    = static_cast<Tref>(0.);
                        dxhathat = static_cast<Tref>(0.);
                        for(int bidx = 0; bidx < n_batchs; bidx++)
                        { // via mini_batch
                            index = in_nstride * bidx + adjIndex;
                            xhat_index =
                                in_cstride * bidx + in_dstride * didx + width * row + column;
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
                            index = in_nstride * bidx + adjIndex;
                            xhat_index =
                                in_cstride * bidx + in_dstride * didx + width * row + column;
                            tmp1          = xhat[xhat_index] * dxhathat + dxhat;
                            tmp2          = n_batchs * dxhat - tmp1;
                            tmp3          = elemInvVar / static_cast<Tref>(n_batchs);
                            dx_ptr[index] = tmp3 * tmp2;
                        } // end for(n_batchs)
                    }     // for (column)
                }         // for (row)
            }             // for (depth)
        }                 // for (channel)
    }                     // end else

    return 0;
}

template <typename Tgpu, typename Tref, typename Tmix>
int miopenBNBwdSpatialRunHost(
    /*      T alpha,
            T beta,
            T alphaParam,
            T betaParam,
    */
    int n_batchs,
    int channels,
    int depth,
    int height,
    int width,
    const Tgpu* x_ptr,  // layer's fwd input
    const Tgpu* dy_ptr, // fwd normalized x
    Tref* dx_ptr,
    Tmix* scale_ptr,
    Tref* dscale_ptr,
    Tref* dbias_ptr,
    Tref epsilon,
    bool savedmeanvar,
    Tref* savedMean,
    Tref* savedInvVariance)
{

    // C*H*W is also stored as in_nstride, H*W is in_cstride, W is in_hstride.
    unsigned int index;
    unsigned int adjIndex;
    unsigned int in_dstride = height * width;
    unsigned int in_cstride = depth * in_dstride;
    unsigned int in_nstride = channels * in_cstride;
    unsigned int Csubindex  = 0;
    Tref elemStd            = static_cast<Tref>(0.);
    Tref mean               = static_cast<Tref>(0.);
    Tref invVar             = static_cast<Tref>(0.);
    Tref dyelem             = static_cast<Tref>(0.);
    Tref NHW                = static_cast<Tref>(n_batchs * in_cstride);

    if(savedmeanvar)
    {
        for(int cidx = 0; cidx < channels; cidx++)
        { // via channel
            Csubindex = in_cstride * cidx;
            for(int didx = 0; didx < depth; didx++)
            { // via depth
                for(int row = 0; row < height; row++)
                { // via rows
                    for(int column = 0; column < width; column++)
                    { // via columns
                        adjIndex = Csubindex + in_dstride * didx + width * row + column;

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
            }

            // process the batch per channel
            for(int didx = 0; didx < depth; didx++)
            { // via depth
                for(int row = 0; row < height; row++)
                { // via rows
                    for(int column = 0; column < width; column++)
                    { // via columns
                        adjIndex = Csubindex + in_dstride * didx + width * row + column;

                        for(int bidx = 0; bidx < n_batchs; bidx++)
                        { // via mini_batch

                            index     = in_nstride * bidx + adjIndex;
                            elemStd   = x_ptr[index] - mean; // (x_i - mean)
                            Tref tmp1 = static_cast<Tref>(NHW) * dy_ptr[index] - dbias_ptr[cidx];
                            Tref tmp2 = -elemStd * invVar * dscale_ptr[cidx];
                            Tref tmp3 = (scale_ptr[cidx] * invVar) / static_cast<Tref>(NHW);
                            dx_ptr[index] = tmp3 * (tmp2 + tmp1);
                        } // end for(n_batchs)
                    }     // for (column)
                }         // for (row)
            }             // for (depth)
        }                 // for (cidx)
    }
    else
    {

#if(MIO_HEIRARCH_SEL == 1)
        Tref variance_accum_arr[MIO_BN_DIST];
        Tref mean_accum_arr[MIO_BN_DIST];
        Tref dbias_accum_arr[MIO_BN_DIST];
        Tref dscale_accum_arr[MIO_BN_DIST];
#else
        std::vector<Tref> xhat(n_batchs * in_cstride);
        unsigned int xhat_index;
#endif

        Tref variance = static_cast<Tref>(0.);
        for(int cidx = 0; cidx < channels; cidx++)
        { // via channel
#if(MIO_HEIRARCH_SEL == 1)
            for(int i = 0; i < MIO_BN_DIST; i++)
            {
                variance_accum_arr[i] = static_cast<Tref>(0.);
                mean_accum_arr[i]     = static_cast<Tref>(0.);
                dbias_accum_arr[i]    = static_cast<Tref>(0.);
                dscale_accum_arr[i]   = static_cast<Tref>(0.);
            }
#endif
            Csubindex = in_cstride * cidx;

            // process the batch per channel
            mean = static_cast<Tref>(0.);
#if(MIO_HEIRARCH_SEL == 0)
            for(int didx = 0; didx < depth; didx++)
            { // via depth
                for(int row = 0; row < height; row++)
                { // via rows
                    for(int column = 0; column < width; column++)
                    { // via columns
                        adjIndex = Csubindex + in_dstride * didx + width * row + column;

                        for(int bidx = 0; bidx < n_batchs; bidx++)
                        { // via mini_batch
                            index = in_nstride * bidx + adjIndex;
                            // #1 calculate the mean
                            // iterating through the stack of images in the mini_batch
                            mean += x_ptr[index];
                        }
                    } // for (column)
                }     // for (row)
            }         // for (depth)
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
            mean /= static_cast<Tref>(NHW);
            // printf("MEAN: %f\n",mean);
            elemStd  = static_cast<Tref>(0.);
            variance = static_cast<Tref>(0.);
#if(MIO_HEIRARCH_SEL == 0)
            for(int didx = 0; didx < depth; didx++)
            { // via depth
                for(int row = 0; row < height; row++)
                { // via rows
                    for(int column = 0; column < width; column++)
                    { // via columns
                        adjIndex = Csubindex + in_dstride * didx + width * row + column;

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
            }                                              // for (depth)
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
            variance /= static_cast<Tref>(NHW); // (1/(N*H*W))*sum{ (x_i - mean)^2 }
            // printf("VARIANCE: %f\n",variance);
            // #3 add epsilon for numeric stability, sqr_root, and invert
            invVar = 1. / sqrt(variance + epsilon);
            // printf("invVar: %f\n",invVar);

            dscale_ptr[cidx] = static_cast<Tref>(0.);
            dbias_ptr[cidx]  = static_cast<Tref>(0.);
#if(MIO_HEIRARCH_SEL == 0)
            for(int didx = 0; didx < depth; didx++)
            { // via depth
                for(int row = 0; row < height; row++)
                { // via rows
                    for(int column = 0; column < width; column++)
                    { // via columns
                        adjIndex = Csubindex + in_dstride * didx + width * row + column;

                        for(int bidx = 0; bidx < n_batchs; bidx++)
                        { // via mini_batch
                            index = in_nstride * bidx + adjIndex;
                            xhat_index =
                                in_cstride * bidx + in_dstride * didx + width * row + column;
                            // per (x-dims) channel load a block of data into LDS
                            elemStd          = x_ptr[index] - mean; // (x_i - mean)
                            xhat[xhat_index] = elemStd * invVar;
                            dyelem           = dy_ptr[index];
                            dbias_ptr[cidx] += dyelem;
                            dscale_ptr[cidx] += xhat[xhat_index] * dyelem;
                        } // end for(n_batchs)
                    }     // for (column)
                }         // for (row)
            }             // for (depth)
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

#if(MIO_HEIRARCH_SEL == 0)
            for(int didx = 0; didx < depth; didx++)
            { // via depth
                for(int row = 0; row < height; row++)
                { // via rows
                    for(int column = 0; column < width; column++)
                    { // via columns
                        adjIndex = Csubindex + in_dstride * didx + width * row + column;
                        for(int bidx = 0; bidx < n_batchs; bidx++)
                        { // via mini_batch
                            index = in_nstride * bidx + adjIndex;
                            xhat_index =
                                in_cstride * bidx + in_dstride * didx + width * row + column;
                            Tref tmp1 = static_cast<Tref>(NHW) * dy_ptr[index] - dbias_ptr[cidx];
                            Tref tmp2 = -xhat[xhat_index] * dscale_ptr[cidx];
                            Tref tmp3 = (scale_ptr[cidx] * invVar) / static_cast<Tref>(NHW);
                            dx_ptr[index] = tmp3 * (tmp2 + tmp1);
                        } // end for(n_batchs)
                    }     // for (column)
                }         // for (row)
            }             // for (depth)
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
                            elemStd   = x_ptr[index] - mean; // (x_i - mean)
                            Tref tmp1 = static_cast<Tref>(NHW) * dy_ptr[index] - dbias_ptr[cidx];
                            Tref tmp2 = -elemStd * invVar * dscale_ptr[cidx];
                            Tref tmp3 = (scale_ptr[cidx] * invVar) / static_cast<Tref>(NHW);
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
