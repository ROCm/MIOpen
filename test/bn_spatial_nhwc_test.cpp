/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

#include "driver.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "test.hpp"
#include "verify.hpp"
#include "random.hpp"
#include <array>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <miopen/batch_norm.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/tensor_layout.hpp>
#include <utility>
#include <cfloat>

#define MIO_BN_TEST_EXPAVGFACTOR 0.1
#define MIO_BN_TEST_EPSILON 1e-5
#define MIO_BN_USE_MIX_PREC 1
#if MIO_BN_USE_MIX_PREC == 1
#define PREC_TYPE float
#else
#define PREC_TYPE T
#endif

template <class T, class U>
struct verify_forward_train_bn_spatial
{
    const tensor<T> input;
    const tensor<U> scale;
    const tensor<U> shift;
    std::tuple<tensor<T>, tensor<U>, tensor<U>, tensor<U>, tensor<U>> cpu() const
    {
        double epsilon      = MIO_BN_TEST_EPSILON;
        double expAvgFactor = MIO_BN_TEST_EXPAVGFACTOR;

        std::size_t n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(input.desc.GetLengths());

        std::size_t rs_n_batch, rs_channels, rs_height, rs_width;
        auto derivedBnDesc =
            miopen::TensorDescriptor(input.desc.GetType(),
                                     std::vector<std::size_t>{1, 1, 1, channels},
                                     std::vector<std::size_t>{channels, channels, channels, 1});
        std::tie(rs_n_batch, rs_height, rs_width, rs_channels) =
            miopen::tien<4>(derivedBnDesc.GetLengths());

        tensor<U> runMean;
        tensor<U> runVar;
        if(input.desc.GetType() == miopenFloat)
        {
            runMean = tensor<U>{rs_n_batch, rs_height, rs_width, rs_channels}.generate(
                tensor_elem_gen_integer{17});
            runVar = tensor<U>{rs_n_batch, rs_height, rs_width, rs_channels}.generate(
                tensor_elem_gen_integer{17});
        }
        else
        {
            srand(0);
            runMean = tensor<U>{rs_n_batch, rs_height, rs_width, rs_channels};
            runVar  = tensor<U>{rs_n_batch, rs_height, rs_width, rs_channels};
            for(std::size_t i = 0; i < runMean.desc.GetElementSize(); i++)
            {
                runMean[i] = (((GET_RAND() % 2) == 1) ? -1 : 1) * 1e-3 * U(GET_RAND() % 100);
                runVar[i]  = 1e-3 * U(GET_RAND() % 100);
            }
        }
        auto saveMean   = tensor<U>{rs_n_batch, rs_height, rs_width, rs_channels};
        auto saveInvVar = tensor<U>{rs_n_batch, rs_height, rs_width, rs_channels};
        auto out        = input;
        std::fill(out.begin(), out.end(), 0);

        const auto nhw = double(height * width * n_batch);
        par_for(channels, 1, [&](int cidx) {
            double elemStd        = 0.;
            double variance_accum = 0.;
            double mean_accum     = 0.;
            double invVar         = 0.;
            double newRunMean     = 0.;
            double adjust         = 0.;

            std::vector<double> variance_accum_arr(height, 0.0);
            std::vector<double> mean_accum_arr(height, 0.0);
            std::vector<double> dshift_accum_arr(height, 0.0);
            std::vector<double> dscale_accum_arr(height, 0.0);

            for(std::size_t row = 0; row < height; row++)
            {
                for(std::size_t column = 0; column < width; column++)
                {
                    for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                    {
                        mean_accum_arr[row] += input(bidx, cidx, row, column);
                    }
                }
            }
            for(std::size_t i = 0; i < height; i++)
                mean_accum += mean_accum_arr[i];

            mean_accum /= nhw;

            elemStd        = 0.;
            variance_accum = 0.;

            for(std::size_t row = 0; row < height; row++)
            {
                for(std::size_t column = 0; column < width; column++)
                {
                    for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                    {
                        out(bidx, cidx, row, column) = elemStd =
                            input(bidx, cidx, row, column) - mean_accum;
                        variance_accum_arr[row] += elemStd * elemStd;
                    }
                }
            }
            for(std::size_t i = 0; i < height; i++)
                variance_accum += variance_accum_arr[i];

            variance_accum /= nhw;
            invVar = 1.0 / sqrt(variance_accum + epsilon);

            for(std::size_t bidx = 0; bidx < n_batch; bidx++)
            {
                for(std::size_t row = 0; row < height; row++)
                {
                    for(std::size_t column = 0; column < width; column++)
                    {
                        out(bidx, cidx, row, column) =
                            scale(0, 0, 0, cidx) * (invVar * out(bidx, cidx, row, column)) +
                            shift(0, 0, 0, cidx);
                    }
                }
            }

            saveMean(0, 0, 0, cidx)   = mean_accum;
            saveInvVar(0, 0, 0, cidx) = invVar;

            newRunMean             = runMean(0, 0, 0, cidx) * (1 - expAvgFactor);
            runMean(0, 0, 0, cidx) = mean_accum * expAvgFactor + newRunMean;
            adjust                 = (n_batch * height * width == 1) ? variance_accum
                                                     : (nhw / (nhw - 1)) * variance_accum;
            runVar(0, 0, 0, cidx) =
                (1 - expAvgFactor) * runVar(0, 0, 0, cidx) + expAvgFactor * adjust;
        });

        return std::make_tuple(out, runMean, runVar, saveMean, saveInvVar);
    }

    std::tuple<tensor<T>, tensor<U>, tensor<U>, tensor<U>, tensor<U>> gpu() const
    {
        auto&& handle = get_handle();

        std::size_t n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(input.desc.GetLengths());

        auto out = input;
        std::fill(out.begin(), out.end(), 0);

        std::size_t rs_n_batch, rs_channels, rs_height, rs_width;
        auto derivedBnDesc =
            miopen::TensorDescriptor(input.desc.GetType(),
                                     std::vector<std::size_t>{1, 1, 1, channels},
                                     std::vector<std::size_t>{channels, channels, channels, 1});
        std::tie(rs_n_batch, rs_height, rs_width, rs_channels) =
            miopen::tien<4>(derivedBnDesc.GetLengths());

        tensor<U> runMean;
        tensor<U> runVar;
        if(input.desc.GetType() == miopenFloat)
        {
            runMean = tensor<U>{rs_n_batch, rs_height, rs_width, rs_channels}.generate(
                tensor_elem_gen_integer{17});
            runVar = tensor<U>{rs_n_batch, rs_height, rs_width, rs_channels}.generate(
                tensor_elem_gen_integer{17});
        }
        else
        {
            srand(0);
            runMean = tensor<U>{rs_n_batch, rs_height, rs_width, rs_channels};
            runVar  = tensor<U>{rs_n_batch, rs_height, rs_width, rs_channels};
            for(std::size_t i = 0; i < runMean.desc.GetElementSize(); i++)
            {
                runMean[i] = (((GET_RAND() % 2) == 1) ? -1 : 1) * 1e-3 * U(GET_RAND() % 100);
                runVar[i]  = 1e-3 * U(GET_RAND() % 100);
            }
        }

        auto saveMean   = tensor<U>{rs_n_batch, rs_height, rs_width, rs_channels};
        auto saveInvVar = tensor<U>{rs_n_batch, rs_height, rs_width, rs_channels};

        auto in_dev    = handle.Write(input.data);
        auto scale_dev = handle.Write(scale.data);
        auto shift_dev = handle.Write(shift.data);

        auto runMean_dev    = handle.Write(runMean.data);
        auto runVar_dev     = handle.Write(runVar.data);
        auto saveMean_dev   = handle.Create<U>(channels);
        auto saveInvVar_dev = handle.Create<U>(channels);
        auto out_dev        = handle.Create<T>(n_batch * channels * height * width);

        double epsilon      = MIO_BN_TEST_EPSILON;
        double expAvgFactor = MIO_BN_TEST_EXPAVGFACTOR;

        float alpha = 1.0;
        float beta  = 0.0;

        miopen::BatchNormForwardTraining(handle,
                                         miopenBNSpatial,
                                         &alpha,
                                         &beta,
                                         input.desc,
                                         in_dev.get(),
                                         out.desc,
                                         out_dev.get(),
                                         scale.desc,
                                         scale_dev.get(),
                                         shift_dev.get(),
                                         expAvgFactor,
                                         runMean_dev.get(),
                                         runVar_dev.get(),
                                         epsilon,
                                         saveMean_dev.get(),
                                         saveInvVar_dev.get());

        saveMean.data   = handle.Read<U>(saveMean_dev, saveMean.data.size());
        saveInvVar.data = handle.Read<U>(saveInvVar_dev, saveInvVar.data.size());
        runMean.data    = handle.Read<U>(runMean_dev, runMean.data.size());
        runVar.data     = handle.Read<U>(runVar_dev, runVar.data.size());
        out.data        = handle.Read<T>(out_dev, out.data.size());

        return std::make_tuple(out, runMean, runVar, saveMean, saveInvVar);
    }

    void fail(int badtensor) const
    {
        std::cout << "Forward Train Spatial Batch Normalization: " << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;

        switch(badtensor)
        {
        case(0): std::cout << "Output tensor output failed verification." << std::endl; break;
        case(1): std::cout << "Running Mean output tensor failed verification." << std::endl; break;
        case(2):
            std::cout << "Running Variance output tensor failed verification." << std::endl;
            break;
        case(3): std::cout << "Saved Mean tensor failed verification." << std::endl; break;
        case(4): std::cout << "Saved Variance tensor failed verification." << std::endl; break;
        default: break;
        }
    }
};

template <class T, class U>
struct verify_backward_bn_spatial_recalc
{
    const tensor<T> x_input;
    const tensor<T> dy_input;
    const tensor<U> scale;

    std::tuple<tensor<T>, tensor<U>, tensor<U>> cpu() const
    {
        double epsilon = MIO_BN_TEST_EPSILON;

        std::size_t n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(x_input.desc.GetLengths());

        std::size_t ss_n_batch, ss_channels, ss_height, ss_width;
        auto derivedBnDesc =
            miopen::TensorDescriptor(x_input.desc.GetType(),
                                     std::vector<std::size_t>{1, 1, 1, channels},
                                     std::vector<std::size_t>{channels, channels, channels, 1});
        std::tie(ss_n_batch, ss_height, ss_width, ss_channels) =
            miopen::tien<4>(derivedBnDesc.GetLengths());

        auto dx_out = dy_input;
        std::fill(dx_out.begin(), dx_out.end(), 0);

        auto dscale = tensor<U>{ss_n_batch, ss_channels, ss_height, ss_width};
        std::fill(dscale.begin(), dscale.end(), 0);

        auto dshift = tensor<U>{ss_n_batch, ss_channels, ss_height, ss_width};
        std::fill(dshift.begin(), dshift.end(), 0);

        const auto nhw = double(height * width * n_batch);

        par_for(channels, 1, [&](int cidx) {
            double elemStd = 0.;
            unsigned int xhat_index;
            double mean     = 0.;
            double invVar   = 0.;
            double dyelem   = 0.;
            double variance = 0.;

            std::vector<double> xhat(height * width * n_batch, 0.0);
            std::vector<double> variance_accum_arr(height, 0.0);
            std::vector<double> mean_accum_arr(height, 0.0);
            std::vector<double> dshift_accum_arr(height, 0.0);
            std::vector<double> dscale_accum_arr(height, 0.0);

            for(std::size_t row = 0; row < height; row++)
            {
                for(std::size_t column = 0; column < width; column++)
                {
                    for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                    {
                        mean_accum_arr[row] += x_input(bidx, cidx, row, column);
                    }
                }
            }
            for(std::size_t i = 0; i < height; i++)
                mean += mean_accum_arr[i];

            mean /= nhw;

            elemStd  = 0.;
            variance = 0.;

            for(std::size_t row = 0; row < height; row++)
            {
                for(std::size_t column = 0; column < width; column++)
                {
                    for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                    {
                        elemStd = x_input(bidx, cidx, row, column) - mean;
                        variance_accum_arr[row] += elemStd * elemStd;
                    }
                }
            }
            for(std::size_t i = 0; i < height; i++)
                variance += variance_accum_arr[i];

            variance /= nhw;
            invVar = 1. / double(sqrt(variance + epsilon));

            dscale(0, cidx, 0, 0) = 0.;

            for(std::size_t row = 0; row < height; row++)
            {
                for(std::size_t column = 0; column < width; column++)
                {
                    for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                    {
                        xhat_index       = height * width * bidx + (width * row + column);
                        elemStd          = x_input(bidx, cidx, row, column) - mean;
                        xhat[xhat_index] = elemStd * invVar;
                        dyelem           = dy_input(bidx, cidx, row, column);
                        dshift_accum_arr[row] += dyelem;
                        dscale_accum_arr[row] += xhat[xhat_index] * dyelem;
                    }
                }
            }
            for(std::size_t i = 0; i < height; i++)
            {
                dshift(0, cidx, 0, 0) += dshift_accum_arr[i];
                dscale(0, cidx, 0, 0) += dscale_accum_arr[i];
            }

            for(std::size_t row = 0; row < height; row++)
            {
                for(std::size_t column = 0; column < width; column++)
                {
                    for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                    {
                        xhat_index = height * width * bidx + (width * row + column);

                        double tmp1 =
                            nhw * dy_input(bidx, cidx, row, column) - dshift(0, cidx, 0, 0);
                        double tmp2                     = -xhat[xhat_index] * dscale(0, cidx, 0, 0);
                        double tmp3                     = (scale(0, 0, 0, cidx) * invVar) / nhw;
                        dx_out(bidx, cidx, row, column) = tmp3 * (tmp2 + tmp1);
                    }
                }
            }
        });

        return std::make_tuple(dx_out, dscale, dshift);
    }

    std::tuple<tensor<T>, tensor<U>, tensor<U>> gpu() const
    {
        auto&& handle = get_handle();

        std::size_t n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(x_input.desc.GetLengths());

        auto dx_out = dy_input;
        std::fill(dx_out.begin(), dx_out.end(), 0);

        std::size_t ss_n_batch, ss_channels, ss_height, ss_width;
        auto derivedBnDesc =
            miopen::TensorDescriptor(x_input.desc.GetType(),
                                     std::vector<std::size_t>{1, 1, 1, channels},
                                     std::vector<std::size_t>{channels, channels, channels, 1});
        std::tie(ss_n_batch, ss_height, ss_width, ss_channels) =
            miopen::tien<4>(derivedBnDesc.GetLengths());

        auto dscale = tensor<U>{ss_n_batch, ss_channels, ss_height, ss_width};
        std::fill(dscale.begin(), dscale.end(), 0);

        auto dshift = tensor<U>{ss_n_batch, ss_channels, ss_height, ss_width};
        std::fill(dshift.begin(), dshift.end(), 0);

        float alpha = 1.0;
        float beta  = 0.0;

        auto xin_dev    = handle.Write(x_input.data);
        auto dyin_dev   = handle.Write(dy_input.data);
        auto scale_dev  = handle.Write(scale.data);
        auto dscale_dev = handle.Write(dscale.data);
        auto dshift_dev = handle.Write(dshift.data);
        auto dx_out_dev = handle.Write(dx_out.data);

        double epsilon = MIO_BN_TEST_EPSILON;

        miopen::BatchNormBackward(handle,
                                  miopenBNSpatial,
                                  &alpha,
                                  &beta,
                                  &alpha,
                                  &beta,
                                  x_input.desc,
                                  xin_dev.get(),
                                  dy_input.desc,
                                  dyin_dev.get(),
                                  dx_out.desc,
                                  dx_out_dev.get(),
                                  scale.desc,
                                  scale_dev.get(),
                                  dscale_dev.get(),
                                  dshift_dev.get(),
                                  epsilon,
                                  nullptr,
                                  nullptr);

        dx_out.data = handle.Read<T>(dx_out_dev, dx_out.data.size());
        dscale.data = handle.Read<U>(dscale_dev, dscale.data.size());
        dshift.data = handle.Read<U>(dshift_dev, dshift.data.size());

        return std::make_tuple(dx_out, dscale, dshift);
    }

    void fail(int badtensor) const
    {
        std::cout << "Backward Batch Spatial Normalization Recalc Mean and Variance: " << std::endl;
        std::cout << "X Input tensor: " << x_input.desc.ToString() << std::endl;
        std::cout << "Delta Y Input tensor: " << dy_input.desc.ToString() << std::endl;
        switch(badtensor)
        {
        case(0):
            std::cout << "Delta X output tensor output failed verification." << std::endl;
            break;
        case(1): std::cout << "Delta scale output tensor failed verification." << std::endl; break;
        case(2): std::cout << "Delta shift output tensor failed verification." << std::endl; break;
        default: break;
        }
    }
};

template <class T, class U>
struct verify_backward_bn_spatial_use_saved
{
    const tensor<T> x_input;
    const tensor<T> dy_input;
    const tensor<U> scale;
    const tensor<U> savedMean;
    const tensor<U> savedInvVar;
    std::tuple<tensor<T>, tensor<U>, tensor<U>> cpu() const
    {

        std::size_t n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(x_input.desc.GetLengths());

        auto dx_out = dy_input;
        std::fill(dx_out.begin(), dx_out.end(), 0);

        std::size_t ss_n_batch, ss_channels, ss_height, ss_width;
        auto derivedBnDesc =
            miopen::TensorDescriptor(x_input.desc.GetType(),
                                     std::vector<std::size_t>{1, 1, 1, channels},
                                     std::vector<std::size_t>{channels, channels, channels, 1});
        std::tie(ss_n_batch, ss_height, ss_width, ss_channels) =
            miopen::tien<4>(derivedBnDesc.GetLengths());

        auto dscale = tensor<U>{ss_n_batch, ss_channels, ss_height, ss_width};
        std::fill(dscale.begin(), dscale.end(), 0);

        auto dshift = tensor<U>{ss_n_batch, ss_channels, ss_height, ss_width};
        std::fill(dshift.begin(), dshift.end(), 0);

        const auto nhw = double(height * width * n_batch);

        par_for(channels, 1, [&](int cidx) {
            double elemStd = 0.;
            unsigned int xhat_index;
            double mean   = savedMean(0, 0, 0, cidx);
            double invVar = savedInvVar(0, 0, 0, cidx);
            double dyelem = 0.;

            std::vector<double> xhat(n_batch * height * width, 0.0);
            std::vector<double> dshift_accum_arr(height, 0.0);
            std::vector<double> dscale_accum_arr(height, 0.0);
            dscale(0, cidx, 0, 0) = 0.;

            for(std::size_t row = 0; row < height; row++)
            {
                for(std::size_t column = 0; column < width; column++)
                {
                    for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                    {
                        xhat_index       = height * width * bidx + (width * row + column);
                        elemStd          = x_input(bidx, cidx, row, column) - mean;
                        xhat[xhat_index] = elemStd * invVar;
                        dyelem           = dy_input(bidx, cidx, row, column);
                        dshift_accum_arr[row] += dyelem;
                        dscale_accum_arr[row] += xhat[xhat_index] * dyelem;
                    }
                }
            }
            for(std::size_t i = 0; i < height; i++)
            {
                dshift(0, cidx, 0, 0) += dshift_accum_arr[i];
                dscale(0, cidx, 0, 0) += dscale_accum_arr[i];
            }

            for(std::size_t row = 0; row < height; row++)
            {
                for(std::size_t column = 0; column < width; column++)
                {
                    for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                    {
                        xhat_index = height * width * bidx + (width * row + column);

                        double tmp1 =
                            nhw * dy_input(bidx, cidx, row, column) - dshift(0, cidx, 0, 0);
                        double tmp2                     = -xhat[xhat_index] * dscale(0, cidx, 0, 0);
                        double tmp3                     = (scale(0, 0, 0, cidx) * invVar) / nhw;
                        dx_out(bidx, cidx, row, column) = tmp3 * (tmp2 + tmp1);
                    }
                }
            }
        });

        return std::make_tuple(dx_out, dscale, dshift);
    }

    std::tuple<tensor<T>, tensor<U>, tensor<U>> gpu() const
    {
        auto&& handle = get_handle();

        std::size_t n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(x_input.desc.GetLengths());

        auto dx_out = dy_input;
        std::fill(dx_out.begin(), dx_out.end(), 0);

        std::size_t ss_n_batch, ss_channels, ss_height, ss_width;
        auto derivedBnDesc =
            miopen::TensorDescriptor(x_input.desc.GetType(),
                                     std::vector<std::size_t>{1, 1, 1, channels},
                                     std::vector<std::size_t>{channels, channels, channels, 1});
        std::tie(ss_n_batch, ss_height, ss_width, ss_channels) =
            miopen::tien<4>(derivedBnDesc.GetLengths());

        auto dscale = tensor<U>{ss_n_batch, ss_channels, ss_height, ss_width};
        std::fill(dscale.begin(), dscale.end(), 0);

        auto dshift = tensor<U>{ss_n_batch, ss_channels, ss_height, ss_width};
        std::fill(dshift.begin(), dshift.end(), 0);

        float alpha = 1.0;
        float beta  = 0.0;

        auto xin_dev         = handle.Write(x_input.data);
        auto dyin_dev        = handle.Write(dy_input.data);
        auto scale_dev       = handle.Write(scale.data);
        auto dscale_dev      = handle.Write(dscale.data);
        auto dshift_dev      = handle.Write(dshift.data);
        auto dx_out_dev      = handle.Write(dx_out.data);
        auto savedMean_dev   = handle.Write(savedMean.data);
        auto savedInvVar_dev = handle.Write(savedInvVar.data);

        double epsilon = MIO_BN_TEST_EPSILON;

        miopen::BatchNormBackward(handle,
                                  miopenBNSpatial,
                                  &alpha,
                                  &beta,
                                  &alpha,
                                  &beta,
                                  x_input.desc,
                                  xin_dev.get(),
                                  dy_input.desc,
                                  dyin_dev.get(),
                                  dx_out.desc,
                                  dx_out_dev.get(),
                                  scale.desc,
                                  scale_dev.get(),
                                  dscale_dev.get(),
                                  dshift_dev.get(),
                                  epsilon,
                                  savedMean_dev.get(),
                                  savedInvVar_dev.get());

        dx_out.data = handle.Read<T>(dx_out_dev, dx_out.data.size());
        dscale.data = handle.Read<U>(dscale_dev, dscale.data.size());
        dshift.data = handle.Read<U>(dshift_dev, dshift.data.size());

        return std::make_tuple(dx_out, dscale, dshift);
    }

    void fail(int badtensor) const
    {
        std::cout << "Backward Batch Spatial Normalization Use Saved Mean and Variance: "
                  << std::endl;
        std::cout << "X Input tensor: " << x_input.desc.ToString() << std::endl;
        std::cout << "Delta Y Input tensor: " << dy_input.desc.ToString() << std::endl;
        switch(badtensor)
        {
        case(0):
            std::cout << "Delta X output tensor output failed verification." << std::endl;
            break;
        case(1): std::cout << "Delta scale output tensor failed verification." << std::endl; break;
        case(2): std::cout << "Delta shift output tensor failed verification." << std::endl; break;
        default: break;
        }
    }
};

template <class T>
struct batch_norm_spatial_nhwc_driver : test_driver
{
    tensor<T> input;
    tensor<PREC_TYPE> scale;
    tensor<PREC_TYPE> shift;
    batch_norm_spatial_nhwc_driver()
    {
        this->batch_factor = 4;
        add(input,
            "input",
            get_bn_spatial_input_tensor(
                tensor_elem_gen_integer{miopen_type<T>{} == miopenHalf ? 5 : 17}));
    }

    void run()
    {
        std::size_t n, c, h, w;
        std::tie(n, c, h, w) = miopen::tien<4>(input.desc.GetLengths());

        std::size_t ssn, ssc, ssh, ssw;
        auto derivedBnDesc           = miopen::TensorDescriptor(input.desc.GetType(),
                                                      std::vector<std::size_t>{1, 1, 1, c},
                                                      std::vector<std::size_t>{c, c, c, 1});
        std::tie(ssn, ssh, ssw, ssc) = miopen::tien<4>(derivedBnDesc.GetLengths());

        std::vector<std::size_t> new_len = input.desc.GetLengths();
        std::vector<std::size_t> new_str;
        miopen::tensor_layout_to_strides(new_len, "NCHW", "NHWC", new_str);
        input.desc = miopen::TensorDescriptor(miopen_type<T>{}, new_len, new_str);

        if(input.desc.GetType() == miopenFloat)
        {
            scale = tensor<PREC_TYPE>{ssn, ssh, ssw, ssc}.generate(tensor_elem_gen_integer{17});
            shift = tensor<PREC_TYPE>{ssn, ssh, ssw, ssc}.generate(tensor_elem_gen_integer{17});
        }
        else
        {
            srand(0);
            scale = tensor<PREC_TYPE>{ssn, ssh, ssw, ssc};
            shift = tensor<PREC_TYPE>{ssn, ssh, ssw, ssc};
            for(std::size_t i = 0; i < scale.desc.GetElementSize(); i++)
            {
                scale[i] = (((GET_RAND() % 2) == 1) ? -1 : 1) * 1e-4 * PREC_TYPE(GET_RAND() % 100);
                shift[i] = (((GET_RAND() % 2) == 1) ? -1 : 1) * 1e-4 * PREC_TYPE(GET_RAND() % 100);
            }
            for(std::size_t i = 0; i < input.desc.GetElementSize(); i++)
            {
                input[i] = (((GET_RAND() % 2) == 1) ? -1 : 1) * (1e-5 * T(GET_RAND() % 100));
            }
        }

        auto outpair = verify(verify_forward_train_bn_spatial<T, PREC_TYPE>{input, scale, shift});

        auto dy_input = std::get<0>(outpair.second);
        for(std::size_t bidx = 0; bidx < n; bidx++)
        {
            for(std::size_t cidx = 0; cidx < c; cidx++)
            {
                for(std::size_t row = 0; row < h; row++)
                {
                    for(std::size_t column = 0; column < w; column++)
                    {
                        dy_input(bidx, cidx, row, column) *= 0.1;
                    }
                }
            }
        }
        this->tolerance = 80 * input.desc.GetElementSize();
        verify(verify_backward_bn_spatial_recalc<T, PREC_TYPE>{input, dy_input, scale});

        auto savedMean   = std::get<3>(outpair.second);
        auto savedInvVar = std::get<4>(outpair.second);
        verify(verify_backward_bn_spatial_use_saved<T, PREC_TYPE>{
            input, dy_input, scale, savedMean, savedInvVar});
    }
};

int main(int argc, const char* argv[])
{
    test_drive<batch_norm_spatial_nhwc_driver>(argc, argv);
    return 0;
}
