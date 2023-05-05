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

#ifndef MIO_CONV_BATCHNORM_ACTIV_HOST_H_
#define MIO_CONV_BATCHNORM_ACTIV_HOST_H_

#include <cmath>
#include <iomanip>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>

template <typename Tgpu, typename Tref>
int miopenBNSpatialFwdInferHost(miopenTensorDescriptor_t& inputTensor,
                                const Tref* in_ptr,
                                Tref* out_ptr,
                                const Tgpu* scale_ptr,
                                const Tgpu* bias_ptr,
                                double epsilon,
                                const Tgpu* estimatedMean,
                                const Tgpu* estimatedVariance)
{
    int nIn, cIn, hIn, wIn;
    miopenGet4dTensorDescriptorLengths(inputTensor, &nIn, &cIn, &hIn, &wIn);

    int n_batchs = nIn;
    int channels = cIn;
    int height   = hIn;
    int width    = wIn;

    unsigned int index;
    unsigned int adjIndex;
    unsigned int in_nstride = channels * height * width;
    unsigned int in_cstride = height * width;

    double elemStd = 0.;
    int ret        = 0;

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
    return (ret);
}

template <typename Tgpu, typename Tref>
int miopenBNPerActivFwdInferHost(miopenTensorDescriptor_t& inputTensor,
                                 const Tref* in_ptr,
                                 Tref* out_ptr,
                                 const Tgpu* scale_ptr,
                                 const Tgpu* bias_ptr,
                                 double epsilon,
                                 const Tgpu* estimatedMean,
                                 const Tgpu* estimatedVariance)
{ // use running mean and variance

    int nIn, cIn, hIn, wIn;
    miopenGet4dTensorDescriptorLengths(inputTensor, &nIn, &cIn, &hIn, &wIn);

    int n_batchs = nIn;
    int channels = cIn;
    int height   = hIn;
    int width    = wIn;

    // C*H*W is also stored as in_nstride, H*W is in_cstride, W is in_hstride.
    unsigned int index;
    unsigned int adjIndex;
    unsigned int in_nstride = channels * height * width;
    unsigned int in_cstride = height * width;

    double elemStd = 0.;

    int ret = 0;

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
                    out_ptr[index] = (scale_ptr[adjIndex] * inhat) + bias_ptr[adjIndex];
                } // end for(n_batchs)
            }     // for (column)
        }
    }
    return (ret);
}

template <typename Tgpu /* the data type used in GPU computations (usually half) */,
          typename Tref /* the data type used in CPU checkings (usually double) */>
void miopenActivationFwdHost(int neuron_type,
                             Tref gamma,
                             Tref beta,
                             Tref alpha,
                             size_t size,
                             const Tref* bot_ptr,
                             Tref* c_res)
{

    /*    std::vector<Tref> data(size, 0.);

        for(size_t k   = 0; k < size; k++)
            data.at(k) = bot_ptr[k];
    */
    std::function<Tref(Tref)> f;

    switch(neuron_type)
    {
    case MIOPEN_NEURON_PASTHRU: //	x
        f = [=](Tref x) { return x; };
        break;
    case MIOPEN_NEURON_LOGISTIC: //	1 / (1 + e^-x)	//Sigmoid
        f = [=](Tref x) { return 1. / (1. + std::exp(Tref(-x))); };
        break;
    case MIOPEN_NEURON_TANH: //	beta * tanh(alpha * x)
        f = [=](Tref x) { return beta * std::tanh(alpha * x); };
        break;
    case MIOPEN_NEURON_RELU: //	max(0, x)
        f = [=](Tref x) { return (x > 0.) ? x : 0.; };
        break;
    case MIOPEN_NEURON_SOFTRELU: //	log(1 + e^x)   // bonomial normal log likelihood
        f = [=](Tref x) {
            return (x > 0.) ? (x + std::log1p(std::exp(-x))) : (std::log1p(std::exp(x)));
        };
        break;
    case MIOPEN_NEURON_ABS: //	abs(x)
        f = [=](Tref x) { return std::abs(x); };
        break;
    case MIOPEN_NEURON_POWER: // (alpha + beta * x) ^ gamma
        f = [=](Tref x) {
            Tref v = alpha + beta * x;
            return v <= std::numeric_limits<Tref>::epsilon() ? 0. : pow(v, gamma);
        };
        break;
    case MIOPEN_NEURON_CLIPPED_RELU: // min(alpha, max(0, x))
        f = [=](Tref x) { return std::min(alpha, std::max(Tref(0.), x)); };
        break;
    case MIOPEN_NEURON_LEAKY_RELU: // alpha * x | x<=0; x | x>0
        f = [=](Tref x) { return (x > 0) ? x : x * alpha; };
        break;
    case MIOPEN_NEURON_ELU: // alpah * (exp(x)-1) | x<=0; x | x>0
        f = [=](Tref x) { return (x > 0) ? x : alpha * std::expm1(x); };
        break;
    default: printf("ERROR: unknown neuron type: %d\n", neuron_type); break;
    }

    for(size_t i = 0; i < size; i++)
        c_res[i] = f(static_cast<Tref>(bot_ptr[i])); // f(data.at(i));
}

template <typename Tgpu /* the data type used in GPU computations (usually half) */,
          typename Tref /* the data type used in CPU checkings (usually double) */>
int miopenInferVerify(size_t size, const Tref* c_res, const Tgpu* top_ptr, Tref allowedEps)
{
    int match = 1;
    for(size_t i = 0; i < size && match; i++)
    {
        Tref c_val     = c_res[i];
        Tref g_val     = static_cast<Tref>(top_ptr[i]);
        double err     = std::abs(c_val - g_val);
        double err_rel = calculate_relative_error(c_val, g_val);
        // if(err > 1e-6) printf("i: %d, cval: %f, gval: %f\n", i, c_val, g_val );
        if((err > allowedEps && err_rel > allowedEps) || std::isnan(c_val) || std::isnan(g_val) ||
           !std::isfinite(c_val) || !std::isfinite(g_val))
        {
            std::cout << "Difference in neuron layer: " << err << " too large at " << i
                      << " c_v = " << c_val << " vs g_val = " << g_val
                      << " tolerance = " << allowedEps << std::endl;
            match = 0;
        }
    }

    return (match);
}

template <typename Tgpu, typename Tref>
int ConvForwardCPU(const std::vector<Tref>& in,
                   std::vector<Tref>& outhost,
                   const std::vector<Tgpu>& wei,
                   const std::vector<Tgpu>& b,
                   const int bias,
                   miopenConvolutionDescriptor_t& convDesc,
                   miopenTensorDescriptor_t& inputTensor,
                   miopenTensorDescriptor_t& weightTensor,
                   miopenTensorDescriptor_t& outputTensor)
{

    int in_n, in_c, in_h, in_w;
    int in_nstride, in_cstride, in_hstride, in_wstride;
    miopenDataType_t dt;
    miopenGet4dTensorDescriptor(inputTensor,
                                &dt,
                                &in_n,
                                &in_c,
                                &in_h,
                                &in_w,
                                &in_nstride,
                                &in_cstride,
                                &in_hstride,
                                &in_wstride);

    int wei_n, wei_c, wei_h, wei_w;
    int wei_nstride, wei_cstride, wei_hstride, wei_wstride;

    miopenGet4dTensorDescriptor(weightTensor,
                                &dt,
                                &wei_n,
                                &wei_c,
                                &wei_h,
                                &wei_w,
                                &wei_nstride,
                                &wei_cstride,
                                &wei_hstride,
                                &wei_wstride);

    int out_n, out_c, out_h, out_w;
    int out_nstride, out_cstride, out_hstride, out_wstride;
    miopenGet4dTensorDescriptor(outputTensor,
                                &dt,
                                &out_n,
                                &out_c,
                                &out_h,
                                &out_w,
                                &out_nstride,
                                &out_cstride,
                                &out_hstride,
                                &out_wstride);

    int stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w;
    miopenConvolutionMode_t mode;
    miopenPaddingMode_t pmode = miopen::deref(convDesc).paddingMode;
    miopenGetConvolutionDescriptor(
        convDesc, &mode, &pad_h, &pad_w, &stride_h, &stride_w, &dilation_h, &dilation_w);

    if(pmode == miopenPaddingSame)
    {
        pad_h = (in_h % stride_h == 0) ? (std::max((wei_h - stride_h), 0))
                                       : (std::max((wei_h - (in_h % stride_h)), 0));
        pad_w = (in_w % stride_w == 0) ? (std::max((wei_w - stride_w), 0))
                                       : (std::max((wei_w - (in_w % stride_w)), 0));
        pad_h /= 2;
        pad_w /= 2;
    }
    else if(pmode == miopenPaddingValid)
    {
        pad_h = 0;
        pad_w = 0;
    }

    if(out_h <= 0 || out_w <= 0)
        throw std::runtime_error("Invalid Test Case: Check Output Dimension.");

    miopenGet4dTensorDescriptor(weightTensor,
                                &dt,
                                &wei_n,
                                &wei_c,
                                &wei_h,
                                &wei_w,
                                &wei_nstride,
                                &wei_cstride,
                                &wei_hstride,
                                &wei_wstride);

    for(int o = 0; o < out_n; o++)
    { // mini-batch size
        for(int w = 0; w < out_c; w++)
        { // out_channels (num filters)
            for(int i = 0; i < out_h; i++)
            { // output_height (from getforwardoutputdim())
                int in_off_h = i * stride_h;
                for(int j = 0; j < out_w; j++)
                { // output_width (from getforwardoutputdim())
                    Tgpu acc     = static_cast<Tgpu>(0.);
                    int in_off_w = j * stride_w;
                    for(int k = 0; k < in_c; k++)
                    { // in_channels (RGB)
                        for(int x = 0; x < wei_h; x++)
                        {
                            int in_x = in_off_h - pad_h + x * dilation_h;
                            if(in_x >= 0 && in_x < in_h)
                            {
                                for(int y = 0; y < wei_w; y++)
                                {
                                    int in_y = in_off_w - pad_w + y * dilation_w;
                                    if(in_y >= 0 && in_y < in_w)
                                    {
                                        acc += static_cast<Tgpu>(in.at(o * in_nstride +
                                                                       k * in_cstride +
                                                                       in_x * in_w + in_y)) *
                                               static_cast<Tgpu>(wei.at(w * wei_nstride +
                                                                        k * wei_cstride +
                                                                        x * wei_hstride + y));
                                    }
                                }
                            }
                        }
                    }
                    acc = bias != 0 ? acc + static_cast<Tgpu>(b[w]) : acc;
                    outhost[o * out_nstride + w * out_cstride + i * out_hstride + j] = acc;
                }
            }
        }
    }

    return 0;
}

#endif
