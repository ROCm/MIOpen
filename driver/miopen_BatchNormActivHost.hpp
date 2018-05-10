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

#ifndef MIO_BATCHNORMACTIVHOST_H_
#define MIO_BATCHNORMACTIVHOST_H_

#include <cmath>
#include <iomanip>

#define MIO_HEIRARCH_SEL 0

#if(MIO_HEIRARCH_SEL == 1)
#define MIO_BN_DIST 32
#endif

template <typename T>
int miopenBNActiveBNSpatialFwdInferHost(int n_batchs,
                                        int channels,
                                        int height,
                                        int width,
                                        const T* in_ptr,
                                        T* out_ptr,
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

template <typename T>
int miopenBNActiveBNPerActivFwdInferHost(int n_batchs,
                                         int channels,
                                         int height,
                                         int width,
                                         const T* in_ptr,
                                         T* out_ptr,
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

template <typename _Tgpu /* the data type used in GPU computations (usually half) */,
          typename _Tcheck /* the data type used in CPU checkings (usually double) */>
int miopenBNActiveNeuronFwdInferHost(int neuron_type,
                                     _Tcheck gamma,
                                     _Tcheck beta,
                                     _Tcheck alpha,
                                     size_t size,
                                     const _Tgpu* bot_ptr,
                                     const _Tgpu* top_ptr,
                                     _Tcheck allowedEps)
{

    int match      = 1;
    _Tcheck* c_res = new _Tcheck[size];
    _Tcheck* data  = new _Tcheck[size];
    for(size_t k = 0; k < size; k++)
        data[k]  = static_cast<_Tcheck>(bot_ptr[k]);

    std::function<_Tcheck(_Tcheck)> f;

    switch(neuron_type)
    {
    case MIOPEN_NEURON_PASTHRU: //	x
        f = [=](_Tcheck x) { return x; };
        break;
    case MIOPEN_NEURON_LOGISTIC: //	1 / (1 + e^-x)	//Sigmoid
        f = [=](_Tcheck x) { return 1 / (1 + std::exp(-x)); };
        break;
    case MIOPEN_NEURON_TANH: //	beta * tanh(alpha * x)
        f = [=](_Tcheck x) { return beta * std::tanh(alpha * x); };
        break;
    case MIOPEN_NEURON_RELU: //	max(0, x)
        f = [=](_Tcheck x) { return (x > 0) ? x : 0; };
        break;
    case MIOPEN_NEURON_SOFTRELU: //	log(1 + e^x)   // bonomial normal log likelihood
        f = [=](_Tcheck x) { return std::log1p(std::exp(x)); };
        break;
    case MIOPEN_NEURON_ABS: //	abs(x)
        f = [=](_Tcheck x) { return std::abs(x); };
        break;
    case MIOPEN_NEURON_POWER: // (alpha + beta * x) ^ gamma
        f = [=](_Tcheck x) {
            _Tcheck v = alpha + beta * x;
            return v <= std::numeric_limits<_Tcheck>::epsilon() ? 0 : pow(v, gamma);
        };
        break;
    case MIOPEN_NEURON_CLIPPED_RELU: // min(alpha, max(0, x))
        f = [=](_Tcheck x) { return std::min(alpha, std::max(_Tcheck(0), x)); };
        break;
    case MIOPEN_NEURON_LEAKY_RELU: // alpha * x | x<=0; x | x>0
        f = [=](_Tcheck x) { return (x > 0) ? x : x * alpha; };
        break;
    case MIOPEN_NEURON_ELU: // alpah * (exp(x)-1) | x<=0; x | x>0
        f = [=](_Tcheck x) { return (x > 0) ? x : alpha * std::expm1(x); };
        break;
    default: printf("ERROR: unknown neuron type: %d\n", neuron_type); break;
    }

    for(size_t i = 0; i < size; i++)
        c_res[i] = f(data[i]);

    for(size_t i = 0; i < size && match; i++)
    {
        _Tcheck c_val  = c_res[i];
        _Tcheck g_val  = static_cast<_Tcheck>(top_ptr[i]);
        double err     = std::abs(c_val - g_val);
        double err_rel = calculate_relative_error(c_val, g_val);

        if((err > allowedEps && err_rel > allowedEps) || std::isnan(c_val) || std::isnan(g_val) ||
           !std::isfinite(c_val) || !std::isfinite(g_val))
        {
            std::cout << "Difference in neuron layer: " << err << " too large at " << i
                      << " x = " << data[i] << " "
                      << " c_v = " << c_val << " vs g_val = " << g_val
                      << " tolerance = " << allowedEps << std::endl;
            match = 0;
        }
    }

    if(c_res)
    {
        delete[] c_res;
    }
    if(data)
    {
        delete[] data;
    }

    return (match);
}

template <typename Tgpu, typename Tref>
int miopenBNActiveVerify(miopenBatchNormMode_t bn_mode,
                         int batch_sz,
                         int channels,
                         int height,
                         int width,
                         const Tgpu* in_ptr,
                         Tgpu* BNout_ptr,
                         Tgpu* scale_ptr,
                         Tgpu* bias_ptr,
                         double epsilon,
                         double* estimatedMean,
                         double* estimatedVariance,
                         int neuron_type,
                         Tref gamma,
                         Tref beta,
                         Tref alpha,
                         size_t size,
                         Tgpu* out_ptr,
                         Tref allowedEps)
{
    if(bn_mode == miopenBNPerActivation)
    { // 1xCxHxW
        miopenBNActiveBNPerActivFwdInferHost(batch_sz,
                                             channels,
                                             height,
                                             width,
                                             in_ptr,
                                             BNout_ptr,
                                             scale_ptr,
                                             bias_ptr,
                                             epsilon,
                                             true,
                                             estimatedMean,
                                             estimatedVariance);
    }
    else if(bn_mode == miopenBNSpatial)
    { // 1xCx1x1
        miopenBNActiveBNSpatialFwdInferHost(/* alpha, beta, */ batch_sz,
                                            channels,
                                            height,
                                            width,
                                            in_ptr,
                                            BNout_ptr,
                                            scale_ptr,
                                            bias_ptr,
                                            epsilon,
                                            true,
                                            estimatedMean,
                                            estimatedVariance);
    }

    int match = miopenBNActiveNeuronFwdInferHost<Tgpu, Tref>(
        neuron_type, gamma, beta, alpha, size, BNout_ptr, out_ptr, allowedEps);

    return match;
}

#endif
