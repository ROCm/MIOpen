/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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
#pragma once
// #include "test.hpp"
#include <array>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <miopen/convolution.hpp>
#include <miopen/batch_norm.hpp>
#include <miopen/activ.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <utility>
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <miopen/fusion_plan.hpp>

template <class T>
void convHostForward(const tensor<T>& input,
                     tensor<T>& output,
                     const tensor<T>& weights,
                     const int bias_mode,
                     const tensor<T>& bias,
                     const miopenConvolutionDescriptor_t convDesc)
{

    int in_n, in_c, in_h, in_w;
    int in_nstride, in_cstride, in_hstride, in_wstride;
    std::tie(in_n, in_c, in_h, in_w) = miopen::tien<4>(input.desc.GetLengths());
    std::tie(in_nstride, in_cstride, in_hstride, in_wstride) =
        miopen::tien<4>(input.desc.GetStrides());

    int wei_n, wei_c, wei_h, wei_w;
    int wei_nstride, wei_cstride, wei_hstride, wei_wstride;
    std::tie(wei_n, wei_c, wei_h, wei_w) = miopen::tien<4>(weights.desc.GetLengths());
    std::tie(wei_nstride, wei_cstride, wei_hstride, wei_wstride) =
        miopen::tien<4>(weights.desc.GetStrides());

    int out_n, out_c, out_h, out_w;
    int out_nstride, out_cstride, out_hstride, out_wstride;
    std::tie(out_n, out_c, out_h, out_w) = miopen::tien<4>(output.desc.GetLengths());
    std::tie(out_nstride, out_cstride, out_hstride, out_wstride) =
        miopen::tien<4>(output.desc.GetStrides());

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
        MIOPEN_THROW("Invalid Test Case: Check Output Dimension.");

    for(int o = 0; o < out_n; o++)
    { // mini-batch size
        for(int w = 0; w < out_c; w++)
        { // out_channels (num filters)
            for(int i = 0; i < out_h; i++)
            { // output_height (from getforwardoutputdim())
                int in_off_h = i * stride_h;
                for(int j = 0; j < out_w; j++)
                { // output_width (from getforwardoutputdim())
                    /*auto acc     = static_cast<T>(0.);*/
                    auto acc     = static_cast<double>(0.);
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
                                        acc += double(
                                            static_cast<T>(input[o * in_nstride + k * in_cstride +
                                                                 in_x * in_w + in_y]) *
                                            static_cast<T>(weights(w, k, x, y)));
                                    }
                                }
                            }
                        }
                    }
                    acc = bias_mode != 0 ? acc + static_cast<double>(bias[w]) : acc;
                    output[o * out_nstride + w * out_cstride + i * out_hstride + j] =
                        static_cast<T>(acc);
                }
            }
        }
    }
}

template <class T, class Tref, class U, class V = U>
void batchNormSpatialHostInference(const tensor<T>& input,
                                   tensor<Tref>& output,
                                   const tensor<U>& scale,
                                   const tensor<U>& bias,
                                   double epsilon,
                                   const tensor<V>& estimatedMean,
                                   const tensor<V>& estimatedVariance)
{

    int n_batches, channels, height, width;
    std::tie(n_batches, channels, height, width) = miopen::tien<4>(input.desc.GetLengths());
    par_for(channels, 1, [&](int cidx) { // via channel
        V mean           = estimatedMean(0, cidx, 0, 0);
        V variance       = estimatedVariance(0, cidx, 0, 0);
        double invertVar = 1.0 / sqrt(variance + epsilon);
        // process the batch per channel
        for(int row = 0; row < height; row++)
        { // via rows
            for(int column = 0; column < width; column++)
            { // via columns
                for(int bidx = 0; bidx < n_batches; bidx++)
                { // via mini_batch
                    double elemStd = static_cast<double>(input(bidx, cidx, row, column)) - mean;
                    double inhat   = elemStd * invertVar;
                    output(bidx, cidx, row, column) =
                        static_cast<T>(scale(0, cidx, 0, 0) * inhat + bias(0, cidx, 0, 0));
                    // printf("output: %f\n",scale(0, cidx, 0, 0) * inhat + bias(0, cidx, 0, 0));
                    // std::cout << output(bidx, cidx, row, column) << ",";
                }
            }
        }
    });
}

template <class T, class U, class V, class Tref>
void batchNormPerActivHostInference(const tensor<T>& input,
                                    tensor<Tref>& output,
                                    const tensor<U>& scale,
                                    const tensor<U>& bias,
                                    double epsilon,
                                    const tensor<V>& estimatedMean,
                                    const tensor<V>& estimatedVariance)
{
    int n_batches, channels, height, width;
    std::tie(n_batches, channels, height, width) = miopen::tien<4>(input.desc.GetLengths());
    par_for(channels, 1, [&](int cidx) { // via channel
        for(int row = 0; row < height; row++)
        { // via rows
            for(int column = 0; column < width; column++)
            { // via columns
                // apply down the n_batch dimension
                double mean       = estimatedMean(0, cidx, row, column);
                double variance   = estimatedVariance(0, cidx, row, column);
                double elemInvVar = 1.0 / sqrt(variance + epsilon);
                for(int bidx = 0; bidx < n_batches; bidx++)
                { // via mini_batch
                    // per (x-dims) channel load a block of data into LDS
                    double elemStd = input(bidx, cidx, row, column) - mean;
                    double inhat   = elemStd * elemInvVar;
                    output(bidx, cidx, row, column) =
                        scale(0, cidx, row, column) * inhat + bias(0, cidx, row, column);
                    //    printf("output: %f\n",output(bidx, cidx, row, column));
                }
            }
        }
    });
}

template <class T, class U, class Tref = U, class Tout>
void batchNormSpatialHostFwdTrain(const tensor<T>& input,
                                  tensor<Tout>& out,
                                  const tensor<U>& scale,
                                  const tensor<U>& bias,
                                  double epsilon,
                                  double expAvgFactor,
                                  tensor<Tref>& saveMean,
                                  tensor<Tref>& saveInvVar,
                                  tensor<Tref>& runMean,
                                  tensor<Tref>& runVar)
{

    int height, width, n_batch, channels;
    std::tie(n_batch, channels, height, width) = miopen::tien<4>(input.desc.GetLengths());
    const auto nhw                             = double(height * width * n_batch);

    par_for(channels, 1, [&](int cidx) {
        double elemStd        = 0.;
        double variance_accum = 0.;
        double mean_accum     = 0.;
        double invVar         = 0.;
        double newRunMean     = 0.;
        double adjust         = 0.;

        // process the batch per channel
        for(int bidx = 0; bidx < n_batch; bidx++)
        { // via mini_batch
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    // #1 calculate the mean
                    // iterating through the stack of images in the mini_batch
                    auto inval = static_cast<double>(input(bidx, cidx, row, column));
                    mean_accum += inval;
                    variance_accum += inval * inval;
                } // end for (column)
            }     // end for (row)
        }         // end for (n)

        mean_accum /= nhw;
        variance_accum /= nhw;
        variance_accum += (-mean_accum * mean_accum);
        invVar = 1.0 / sqrt(variance_accum + epsilon);

        // #4 apply the normalization
        // x_hat = (x_i - mean) / sqrt(variance_accum + epsilon)
        for(int bidx = 0; bidx < n_batch; bidx++)
        { // via mini_batch
            for(int row = 0; row < height; row++)
            { // via rows
                for(int column = 0; column < width; column++)
                { // via columns
                    // #5 Gamma and Beta adjust
                    // y_i = gamma*x_hat + beta
                    elemStd = (static_cast<double>(input(bidx, cidx, row, column)) -
                               mean_accum); // (x_i - mean)
                    out(bidx, cidx, row, column) = static_cast<T>(
                        scale(0, cidx, 0, 0) * (invVar * elemStd) + bias(0, cidx, 0, 0));
                } // for (column)
            }     // for (row)
        }         // end for(n_batchs)
        if(!saveMean.data.empty())
        {
            saveMean(0, cidx, 0, 0)   = mean_accum;
            saveInvVar(0, cidx, 0, 0) = invVar;
        }
        if(!runMean.data.empty())
        {
            newRunMean             = runMean(0, cidx, 0, 0) * (1 - expAvgFactor);
            runMean(0, cidx, 0, 0) = mean_accum * expAvgFactor + newRunMean; // newMean*factor + tmp
            // var(n+1) = p * var(n-1) + (1 - p)*(b/b-1)*var(n)
            adjust = (n_batch * height * width == 1) ? variance_accum
                                                     : (nhw / (nhw - 1)) * variance_accum;
            runVar(0, cidx, 0, 0) =
                (1 - expAvgFactor) * runVar(0, cidx, 0, 0) + expAvgFactor * adjust;
        }
    });
}

template <typename XDataType,
          typename DxDataType,
          typename DyDataType,
          typename ScaleDataType,
          typename AccDataType,
          typename RefDataType>
void batchNormSpatialHostBwdTrain(const tensor<XDataType>& x_input,
                                  const tensor<DyDataType>& dy_input,
                                  tensor<DxDataType>& dx_out,
                                  const tensor<ScaleDataType>& bnScale,
                                  tensor<RefDataType>& dscale,
                                  tensor<RefDataType>& dbias,
                                  const tensor<AccDataType>& savedMean,
                                  const tensor<AccDataType>& savedInvVar)
{
    int height, width, n_batch, channels;
    std::tie(n_batch, channels, height, width) = miopen::tien<4>(x_input.desc.GetLengths());
    auto nhw                                   = double(height * width * n_batch);
    int in_cstride                             = height * width;

    par_for(channels, 1, [&](int cidx) {
        double elemStd = 0.;
        unsigned int xhat_index;
        double mean   = 0.0;
        double invVar = 0.0;
        double dyelem = 0.;
        std::vector<double> xhat(static_cast<std::size_t>(n_batch) * in_cstride, 0.0);
        // process the batch per channel
        dscale(0, cidx, 0, 0) = 0.;
        dbias(0, cidx, 0, 0)  = 0.;

        if(!savedMean.data.empty())
        {

            mean   = savedMean(0, cidx, 0, 0);   // HxW elements
            invVar = savedInvVar(0, cidx, 0, 0); // HxW elements
        }
        else
        {
            double variance_accum = 0.;
            double mean_accum     = 0.;
            double inv_Var        = 0.;

            // process the batch per channel
            for(int bidx = 0; bidx < n_batch; bidx++)
            { // via mini_batch
                for(int row = 0; row < height; row++)
                { // via rows
                    for(int column = 0; column < width; column++)
                    { // via columns
                        // #1 calculate the mean
                        // iterating through the stack of images in the mini_batch
                        auto inval = static_cast<double>(x_input(bidx, cidx, row, column));
                        mean_accum += inval;
                        variance_accum += inval * inval;
                    } // end for (column)
                }     // end for (row)
            }         // end for (n)

            mean_accum /= nhw;
            variance_accum /= nhw;
            variance_accum += (-mean_accum * mean_accum);
            inv_Var = 1.0 / sqrt(variance_accum);

            mean   = mean_accum;
            invVar = inv_Var;
        }
        for(int row = 0; row < height; row++)
        { // via rows
            for(int column = 0; column < width; column++)
            { // via columns
                for(int bidx = 0; bidx < n_batch; bidx++)
                { // via mini_batch
                    xhat_index = in_cstride * bidx + (width * row + column);
                    // per (x-dims) channel load a block of data into LDS
                    elemStd = static_cast<double>(x_input(bidx, cidx, row, column)) -
                              mean; // (x_i - mean)
                    xhat[xhat_index] = elemStd * invVar;
                    dyelem           = static_cast<double>(dy_input(bidx, cidx, row, column));
                    dbias(0, cidx, 0, 0) += dyelem;
                    dscale(0, cidx, 0, 0) += xhat[xhat_index] * dyelem;
                } // end for(n_batch)
            }     // for (column)
        }         // for (row)

        for(int row = 0; row < height; row++)
        { // via rows
            for(int column = 0; column < width; column++)
            { // via columns
                for(int bidx = 0; bidx < n_batch; bidx++)
                { // via mini_batch
                    xhat_index = in_cstride * bidx + (width * row + column);

                    double tmp1 = nhw * dy_input(bidx, cidx, row, column) - dbias(0, cidx, 0, 0);
                    double tmp2 = -xhat[xhat_index] * dscale(0, cidx, 0, 0);
                    double tmp3 = (bnScale(0, cidx, 0, 0) * invVar) / nhw;
                    dx_out(bidx, cidx, row, column) =
                        static_cast<RefDataType>(tmp3 * (tmp2 + tmp1));
                } // end for(n_batchs)
            }     // for (column)
        }         // for (row)
    });           // for (channel)
}

template <typename XDataType,
          typename DyDataType,
          typename DxDataType,
          typename ScaleDataType,
          typename AccDataType,
          typename OutRefDataType,
          typename RefDataType>
void batchNormActivSpatialHostBwdTrain(miopenActivationMode_t activMode,
                                       double gamma,
                                       double beta,
                                       double alpha,
                                       const tensor<XDataType>& x_input,
                                       const tensor<DyDataType>& dy_input,
                                       const tensor<DxDataType>& y_input,
                                       tensor<OutRefDataType>& dx_out,
                                       const tensor<ScaleDataType>& bnScale,
                                       const tensor<AccDataType>& bias,
                                       tensor<RefDataType>& dscale,
                                       tensor<RefDataType>& dbias,
                                       const tensor<AccDataType>& savedMean,
                                       const tensor<AccDataType>& savedInvVar)
{

    int height, width, n_batch, channels;
    std::tie(n_batch, channels, height, width) = miopen::tien<4>(x_input.desc.GetLengths());
    auto nhw                                   = double(height * width * n_batch);
    int in_cstride                             = height * width;

    par_for(channels, 1, [&](int cidx) {
        double elemStd = 0.;
        unsigned int xhat_index;
        double mean   = static_cast<double>(savedMean(0, cidx, 0, 0));   // HxW elements
        double invVar = static_cast<double>(savedInvVar(0, cidx, 0, 0)); // HxW elements
        double dyelem = 0.;
        std::vector<double> xhat(static_cast<std::size_t>(n_batch) * in_cstride, 0.0);
        // process the batch per channel
        dscale(0, cidx, 0, 0) = 0.;
        dbias(0, cidx, 0, 0)  = 0.;

        for(int row = 0; row < height; row++)
        { // via rows
            for(int column = 0; column < width; column++)
            { // via columns
                for(int bidx = 0; bidx < n_batch; bidx++)
                { // via mini_batch

                    // recompute forward batch norm
                    xhat_index = in_cstride * bidx + (width * row + column);
                    // per (x-dims) channel load a block of data into LDS
                    elemStd = static_cast<double>(x_input(bidx, cidx, row, column)) -
                              mean; // (x_i - mean)
                    xhat[xhat_index] = elemStd * invVar;
                    double bnrefowd =
                        bnScale(0, cidx, 0, 0) * xhat[xhat_index] + bias(0, cidx, 0, 0);
                    activationHostBwdElement(activMode,
                                             gamma,
                                             beta,
                                             alpha,
                                             dy_input(bidx, cidx, row, column),
                                             bnrefowd,
                                             y_input(bidx, cidx, row, column),
                                             dyelem);
                    dbias(0, cidx, 0, 0) += dyelem;
                    dscale(0, cidx, 0, 0) += xhat[xhat_index] * dyelem;
                } // end for(n_batch)
            }     // for (column)
        }         // for (row)

        for(int row = 0; row < height; row++)
        { // via rows
            for(int column = 0; column < width; column++)
            { // via columns
                for(int bidx = 0; bidx < n_batch; bidx++)
                { // via mini_batch
                    xhat_index = in_cstride * bidx + (width * row + column);
                    double bnrefowd =
                        bnScale(0, cidx, 0, 0) * xhat[xhat_index] + bias(0, cidx, 0, 0);
                    activationHostBwdElement(activMode,
                                             gamma,
                                             beta,
                                             alpha,
                                             dy_input(bidx, cidx, row, column),
                                             bnrefowd,
                                             y_input(bidx, cidx, row, column),
                                             dyelem);
                    // double tmp1 = nhw * dy_input(bidx, cidx, row, column) - dbias(0, cidx, 0, 0);
                    double tmp1                     = nhw * dyelem - dbias(0, cidx, 0, 0);
                    double tmp2                     = -xhat[xhat_index] * dscale(0, cidx, 0, 0);
                    double tmp3                     = (bnScale(0, cidx, 0, 0) * invVar) / nhw;
                    dx_out(bidx, cidx, row, column) = static_cast<DxDataType>(tmp3 * (tmp2 + tmp1));
                } // end for(n_batchs)
            }     // for (column)
        }         // for (row)
    });           // for (channel)
}

template <class T, class U, class Tref, class TOutref>
void batchNormPerActHostFwdTrain(const tensor<T>& input,
                                 tensor<TOutref>& out,
                                 const tensor<U>& scale,
                                 const tensor<U>& bias,
                                 double epsilon,
                                 double expAvgFactor,
                                 tensor<Tref>& saveMean,
                                 tensor<Tref>& saveInvVar,
                                 tensor<Tref>& runMean,
                                 tensor<Tref>& runVar)
{

    int height, width, n_batch, channels;
    std::tie(n_batch, channels, height, width) = miopen::tien<4>(input.desc.GetLengths());
    const auto n                               = double(n_batch);

    par_for(channels, 1, [&](int cidx) {
        double mean_accum     = 0.;
        double variance_accum = 0.;
        double elemStd        = 0.;
        double elemInvVar     = 0.;
        double inhat          = 0.;
        double newRunMean     = 0.;
        double adjust         = 0.;

        // process the batch per channel
        for(int row = 0; row < height; row++)
        { // via rows
            for(int column = 0; column < width; column++)
            { // via columns

                mean_accum     = 0.;
                variance_accum = 0.;
                for(int bidx = 0; bidx < n_batch; bidx++)
                { // via mini_batch
                    // #1 calculate the mean :: iterating through the stack of images in the
                    // mini_batch
                    auto intval = static_cast<double>(input(bidx, cidx, row, column));
                    mean_accum += intval;
                    variance_accum += intval * intval;
                }
                mean_accum /= n;
                variance_accum /= n;
                variance_accum = variance_accum - (mean_accum * mean_accum);
                elemInvVar     = 1.0 / double(sqrt(variance_accum + epsilon));

                // #4 apply the normalization :: x_hat = (x_i - mean) / sqrt(variance_accum -
                // epsilon)
                for(int bidx = 0; bidx < n_batch; bidx++)
                {                                                            // via mini_batch
                    elemStd = (input(bidx, cidx, row, column) - mean_accum); // (x_i - mean)
                    inhat   = elemStd * elemInvVar;
                    // #5 Gamma and Beta adjust :: y_i = gamma*x_hat + beta
                    out(bidx, cidx, row, column) = static_cast<Tref>(
                        scale(0, cidx, row, column) * inhat + bias(0, cidx, row, column));
                } // end for(n_batch)

                newRunMean = runMean(0, cidx, row, column) * (1.0 - expAvgFactor);
                runMean(0, cidx, row, column) =
                    mean_accum * expAvgFactor + newRunMean; // newMean*factor + tmp

                // var(n+1) = p * var(n-1) + (1 - p)*(b/b-1)*var(n)
                adjust = (n_batch == 1) ? variance_accum : (n / (n - 1.0)) * variance_accum;
                runVar(0, cidx, row, column) =
                    (1 - expAvgFactor) * runVar(0, cidx, row, column) + expAvgFactor * adjust;

                saveMean(0, cidx, row, column)   = static_cast<Tref>(mean_accum);
                saveInvVar(0, cidx, row, column) = static_cast<Tref>(elemInvVar);

            } // for (column)
        }     // for (row)
    });
}

template <class T, class U, class Tref>
void batchNormPerActHostBwdTrain(const tensor<T>& x_input,
                                 const tensor<T>& dy_input,
                                 const tensor<U>& scale,
                                 tensor<Tref>& dscale,
                                 tensor<Tref>& dbias,
                                 tensor<Tref>& dx_out,
                                 const tensor<U>& savedMean,
                                 const tensor<U>& savedInvVar)
{

    int height, width, n_batch, channels;
    std::tie(n_batch, channels, height, width) = miopen::tien<4>(x_input.desc.GetLengths());
    int in_cstride                             = height * width;
    auto n                                     = double(n_batch);

    par_for(channels, 1, [&](int cidx) {
        double elemStd = 0.;
        unsigned int xhat_index;
        double mean       = 0.;
        double elemInvVar = 0.;
        double dyelem     = 0.;
        double dxhat      = 0.;
        double dxhathat   = 0.;
        double tmp1       = 0.;
        std::vector<double> xhat(static_cast<std::size_t>(n_batch) * in_cstride);

        // process the batch per channel
        for(int row = 0; row < height; row++)
        { // via rows
            for(int column = 0; column < width; column++)
            { // via columns
                dxhat    = 0.;
                dxhathat = 0.;

                mean       = savedMean(0, cidx, row, column);   // HxW elements
                elemInvVar = savedInvVar(0, cidx, row, column); // HxW elements

                for(int bidx = 0; bidx < n_batch; bidx++)
                { // via mini_batch
                    xhat_index = in_cstride * bidx + (width * row + column);
                    // per (x-dims) channel load a block of data into LDS
                    elemStd = static_cast<double>(x_input(bidx, cidx, row, column)) -
                              mean; // (x_i - mean)
                    xhat[xhat_index] = elemStd * elemInvVar;
                    dyelem           = static_cast<double>(dy_input(bidx, cidx, row, column));
                    dbias(0, cidx, row, column) += dyelem;
                    dscale(0, cidx, row, column) += xhat[xhat_index] * dyelem;
                    tmp1 = scale(0, cidx, row, column) * dyelem;
                    dxhat += tmp1;
                    dxhathat += tmp1 * xhat[xhat_index];

                } // end for(n_batchs)

                for(int bidx = 0; bidx < n_batch; bidx++)
                { // via mini_batch
                    xhat_index = in_cstride * bidx + (width * row + column);
                    tmp1       = xhat[xhat_index] * dxhathat + dxhat;
                    double tmp2 =
                        n_batch * scale(0, cidx, row, column) * dy_input(bidx, cidx, row, column) -
                        tmp1;
                    double tmp3                     = elemInvVar / (double(n));
                    dx_out(bidx, cidx, row, column) = static_cast<T>(tmp3 * tmp2);
                } // end for(n_batchs)
            }     // for (column)
        }         // for (row)
    });
}

template <class T, class U>
void batchNormActivPerActHostBwdTrain(miopenActivationMode_t activMode,
                                      double gamma,
                                      double beta,
                                      double alpha,
                                      const tensor<T>& x_input,
                                      const tensor<T>& dy_input,
                                      const tensor<T>& y_input,
                                      tensor<T>& dx_out,
                                      const tensor<U>& scale,
                                      const tensor<U>& bias,
                                      tensor<U>& dscale,
                                      tensor<U>& dbias,
                                      const tensor<U>& savedMean,
                                      const tensor<U>& savedInvVar)
{

    int height, width, n_batch, channels;
    std::tie(n_batch, channels, height, width) = miopen::tien<4>(x_input.desc.GetLengths());
    int in_cstride                             = height * width;
    auto n                                     = double(n_batch);

    par_for(channels, 1, [&](int cidx) {
        double elemStd = 0.;
        unsigned int xhat_index;
        double mean       = 0.;
        double elemInvVar = 0.;
        double dyelem     = 0.;
        double dxhat      = 0.;
        double dxhathat   = 0.;
        double tmp1       = 0.;
        std::vector<double> xhat(static_cast<std::size_t>(n_batch) * in_cstride);

        // process the batch per channel
        for(int row = 0; row < height; row++)
        { // via rows
            for(int column = 0; column < width; column++)
            { // via columns
                dxhat    = 0.;
                dxhathat = 0.;

                mean       = savedMean(0, cidx, row, column);   // HxW elements
                elemInvVar = savedInvVar(0, cidx, row, column); // HxW elements

                for(int bidx = 0; bidx < n_batch; bidx++)
                { // via mini_batch
                    xhat_index = in_cstride * bidx + (width * row + column);
                    // per (x-dims) channel load a block of data into LDS
                    elemStd = static_cast<double>(x_input(bidx, cidx, row, column)) -
                              mean; // (x_i - mean)
                    xhat[xhat_index] = elemStd * elemInvVar;
                    double bnrefowd =
                        scale(0, cidx, row, column) * xhat[xhat_index] + bias(0, cidx, row, column);
                    activationHostBwdElement(activMode,
                                             gamma,
                                             beta,
                                             alpha,
                                             dy_input(bidx, cidx, row, column),
                                             bnrefowd,
                                             y_input(bidx, cidx, row, column),
                                             dyelem);
                    /*dyelem           = static_cast<double>(dy_input(bidx, cidx, row, column));*/
                    dbias(0, cidx, row, column) += dyelem;
                    dscale(0, cidx, row, column) += xhat[xhat_index] * dyelem;
                    tmp1 = scale(0, cidx, row, column) * dyelem;
                    dxhat += tmp1;
                    dxhathat += tmp1 * xhat[xhat_index];

                } // end for(n_batchs)

                for(int bidx = 0; bidx < n_batch; bidx++)
                { // via mini_batch
                    xhat_index = in_cstride * bidx + (width * row + column);
                    tmp1       = xhat[xhat_index] * dxhathat + dxhat;
                    double bnrefowd =
                        scale(0, cidx, row, column) * xhat[xhat_index] + bias(0, cidx, row, column);
                    activationHostBwdElement(activMode,
                                             gamma,
                                             beta,
                                             alpha,
                                             dy_input(bidx, cidx, row, column),
                                             bnrefowd,
                                             y_input(bidx, cidx, row, column),
                                             dyelem);
                    double tmp2 = (n_batch * scale(0, cidx, row, column) * dyelem) - tmp1;
                    double tmp3 = elemInvVar / (double(n));
                    dx_out(bidx, cidx, row, column) = static_cast<T>(tmp3 * tmp2);
                } // end for(n_batchs)
            }     // for (column)
        }         // for (row)
    });
}

template <class F>
void visitActivationHostInfer(
    miopenActivationMode_t activMode, double gamma, double beta, double alpha, F f)
{
    switch(activMode)
    {
    case miopenActivationPASTHRU: //  x
        f([=](double x) { return x; });
        break;
    case miopenActivationLOGISTIC: // 1 / (1 + e^-x)  //Sigmoid
        f([=](double x) { return (1. / (1. + std::exp(-x))); });
        break;
    case miopenActivationTANH: // beta * tanh(alpha * x)
        f([=](double x) { return (beta * std::tanh(alpha * x)); });
        break;
    case miopenActivationRELU: // max(0, x)
        f([=](double x) { return ((x > 0.) ? x : 0.); });
        break;
    case miopenActivationSOFTRELU: //  log(1 + e^x)   // bonomial normal log likelihood
        f([=](double x) {
            return (x > 0.) ? (x + std::log1p(std::exp(-x))) : (std::log1p(std::exp(x)));
        });
        break;
    case miopenActivationABS: //  abs(x)
        f([=](double x) { return (std::fabs(x)); });
        break;
    case miopenActivationPOWER: // (alpha + beta * x) ^ gamma
        f([=](double x) {
            auto v = (alpha + beta * x);
            return (v <= std::numeric_limits<double>::epsilon()) ? 0. : pow(v, gamma);
        });
        break;
    case miopenActivationCLIPPEDRELU: // min(alpha, max(0, x))
        f([=](double x) { return (std::min(alpha, std::max(double(0.), x))); });
        break;
    case miopenActivationLEAKYRELU: // alpha * x | x<=0; x | x>0
        f([=](double x) { return ((x > 0.) ? x : x * alpha); });
        break;
    case miopenActivationELU: // alpah * (exp(x)-1) | x<=0; x | x>0
        f([=](double x) { return ((x > 0.) ? x : alpha * std::expm1(x)); });
        break;
        // default: printf("ERROR: unknown neuron type: %d\n", activMode); break;
    }
}

template <class T>
inline void activationHostInfer(miopenActivationMode_t activMode,
                                double gamma,
                                double beta,
                                double alpha,
                                const std::vector<T> input,
                                std::vector<T>& output)
{
    visitActivationHostInfer(activMode, gamma, beta, alpha, [&](auto f) {
        par_for(input.size(), 1, [&](int index) {
            output[index] = static_cast<T>(f(static_cast<double>(input[index])));
        });
    });
}

template <class F>
void visitActivationHostBwd(
    miopenActivationMode_t activMode, double gamma, double beta, double alpha, F f)
{
    switch(activMode)
    {
    case miopenActivationPASTHRU: //  x
        f([=](double dy, double, double) { return dy; });
        break;
    case miopenActivationLOGISTIC: // 1 / (1 + e^-x)  //Sigmoid
        f([=](double dy, double, double y) { return dy * y * (1 - y); });
        break;
    case miopenActivationTANH: // beta * tanh(alpha * x)
        f([=](double dy, double, double y) { return dy * alpha * (beta - y * y / beta); });
        break;
    case miopenActivationRELU: // max(0, x)
        f([=](double dy, double x, double) { return (x > 0) ? dy : 0; });
        break;
    case miopenActivationSOFTRELU: //  log(1 + e^x)   // bonomial normal log likelihood
        f([=](double dy, double x, double) {
            static const double threshold = 50.;
            double expval                 = std::exp(std::min(x, threshold));
            return dy * expval / (expval + 1.0);
        });
        break;
    case miopenActivationABS: //  abs(x)
        f([=](double dy, double x, double) { return dy * ((x > 0) ? 1 : -1); });
        break;
    case miopenActivationPOWER: // (alpha + beta * x) ^ gamma
        f([=](double, double x, double y) {
            auto v = alpha + beta * x;
            return v <= std::numeric_limits<double>::epsilon() ? 0 : gamma * beta * y / v;
        });
        break;
    case miopenActivationCLIPPEDRELU: // min(alpha, max(0, x))
        f([=](double dy, double x, double) { return (x > 0 && x <= alpha) ? dy : 0; });
        break;
    case miopenActivationLEAKYRELU: // alpha * x | x<=0; x | x>0
        f([=](double dy, double x, double) { return dy * ((x > 0) ? 1 : alpha); });
        break;
    case miopenActivationELU: // alpah * (exp(x)-1) | x<=0; x | x>0
        f([=](double dy, double x, double y) { return dy * ((x > 0) ? 1 : y + alpha); });
        break;
        // default: printf("ERROR: unknown neuron type: %d\n", activMode); break;
    }
}

template <class T>
inline void activationHostBwd(miopenActivationMode_t activMode,
                              double gamma,
                              double beta,
                              double alpha,
                              const std::vector<T> dyinput,
                              const std::vector<T> xinput,
                              const std::vector<T> yinput,
                              std::vector<T>& output)
{
    visitActivationHostBwd(activMode, gamma, beta, alpha, [&](auto f) {
        par_for(dyinput.size(), 1, [&](int index) {
            output[index] = static_cast<T>(f(static_cast<double>(dyinput[index]),
                                             static_cast<double>(xinput[index]),
                                             static_cast<double>(yinput[index])));
        });
    });
}

inline void activationHostBwdElement(miopenActivationMode_t activMode,
                                     double gamma,
                                     double beta,
                                     double alpha,
                                     const double dyinput,
                                     const double xinput,
                                     const double yinput,
                                     double& output)
{
    visitActivationHostBwd(activMode, gamma, beta, alpha, [&](auto f) {
        output = static_cast<double>(f(dyinput, xinput, yinput));
    });
}

template <class T>
tensor<T> get_output_tensor(const miopen::ConvolutionDescriptor& filter,
                            const tensor<T>& input,
                            const tensor<T>& weights)
{
    return tensor<T>{filter.GetForwardOutputTensor(input.desc, weights.desc, miopen_type<T>{})};
}
