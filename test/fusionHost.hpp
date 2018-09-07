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
#pragma once
#include "test.hpp"
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
#include "driver.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

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

    int u, v, pad_h, pad_w, dilation_h, dilation_w;
    miopenConvolutionMode_t mode;
    miopenPaddingMode_t pmode = miopen::deref(convDesc).paddingMode;
    miopenGetConvolutionDescriptor(
        convDesc, &mode, &pad_h, &pad_w, &u, &v, &dilation_h, &dilation_w);

    if(pmode == miopenPaddingSame)
    {
        pad_h = (in_h % u == 0) ? (std::max((wei_h - u), 0)) : (std::max((wei_h - (in_h % u)), 0));
        pad_w = (in_w % v == 0) ? (std::max((wei_w - v), 0)) : (std::max((wei_w - (in_w % v)), 0));
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
                int in_off_h = i * u;
                for(int j = 0; j < out_w; j++)
                { // output_width (from getforwardoutputdim())
                    /*auto acc     = static_cast<T>(0.);*/
                    auto acc     = static_cast<double>(0.);
                    int in_off_w = j * v;
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

template <class T>
void batchNormSpatialHostInference(const tensor<T>& input,
                                   tensor<T>& output,
                                   const tensor<T>& scale,
                                   const tensor<T>& bias,
                                   double epsilon,
                                   const tensor<T>& estimatedMean,
                                   const tensor<T>& estimatedVariance)
{

    int n_batches, channels, height, width;
    std::tie(n_batches, channels, height, width) = miopen::tien<4>(input.desc.GetLengths());
    par_for(channels, 1, [&](int cidx) { // via channel
        double mean      = estimatedMean(0, cidx, 0, 0);
        double variance  = estimatedVariance(0, cidx, 0, 0);
        double invertVar = 1.0 / sqrt(variance + epsilon);
        // process the batch per channel
        for(int row = 0; row < height; row++)
        { // via rows
            for(int column = 0; column < width; column++)
            { // via columns
                for(int bidx = 0; bidx < n_batches; bidx++)
                { // via mini_batch
                    double elemStd = input(bidx, cidx, row, column) - mean;
                    double inhat   = elemStd * invertVar;
                    output(bidx, cidx, row, column) =
                        scale(0, cidx, 0, 0) * inhat + bias(0, cidx, 0, 0);
                }
            }
        }
    });
}

template <class T>
void batchNormPerActivHostInference(const tensor<T>& input,
                                    tensor<T>& output,
                                    const tensor<T>& scale,
                                    const tensor<T>& bias,
                                    double epsilon,
                                    const tensor<T>& estimatedMean,
                                    const tensor<T>& estimatedVariance)
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
                }
            }
        }
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
void activationHostInfer(miopenActivationMode_t activMode,
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

template <class T>
tensor<T> get_output_tensor(const miopen::ConvolutionDescriptor& filter,
                            const tensor<T>& input,
                            const tensor<T>& weights)
{
    return tensor<T>{filter.GetForwardOutputTensor(input.desc, weights.desc)};
}
