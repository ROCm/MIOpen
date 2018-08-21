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
