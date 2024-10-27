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
#include <utility>
#include <cfloat>
// Run CPU emulations in hierarchical reduction mode.
#define MIO_HEIRARCH_SEL 1
#define MIO_BN_TEST_EXPAVGFACTOR 0.1
#define MIO_BN_TEST_EPSILON 1e-5 // FLT_EPSILON
#define MIO_BN_SP_TEST_DEBUG 0
#define MIO_BN_USE_MIX_PREC 1
#if MIO_BN_USE_MIX_PREC == 1
#define PREC_TYPE float
#else
#define PREC_TYPE T
#endif

//****************************************************
// FORWARD TRAIN
//****************************************************
template <class T, class U>
struct verify_forward_train_3d_bn_spatial
{

    const tensor<T> input;
    const tensor<U> scale;
    const tensor<U> shift;

    std::tuple<tensor<T>, tensor<U>, tensor<U>, tensor<U>, tensor<U>> cpu() const
    {

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        double epsilon      = MIO_BN_TEST_EPSILON;
        double expAvgFactor = MIO_BN_TEST_EXPAVGFACTOR;

        std::size_t n_batch, channels, depth, height, width;
        std::tie(n_batch, channels, depth, height, width) =
            miopen::tien<5>(input.desc.GetLengths());

        std::size_t rs_n_batch, rs_channels, rs_depth, rs_height, rs_width;
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, input.desc, miopenBNSpatial);

        std::tie(rs_n_batch, rs_channels, rs_depth, rs_height, rs_width) =
            miopen::tien<5>(derivedBnDesc.GetLengths());

        tensor<U> runMean;
        tensor<U> runVar;

        if(input.desc.GetType() == miopenFloat)
        {
            runMean = tensor<U>{rs_n_batch, rs_channels, rs_depth, rs_height, rs_width}.generate(
                tensor_elem_gen_integer{17});
            runVar = tensor<U>{rs_n_batch, rs_channels, rs_depth, rs_height, rs_width}.generate(
                tensor_elem_gen_integer{17});
        }
        else
        {
            prng::reset_seed();
            runMean = tensor<U>{rs_n_batch, rs_channels, rs_depth, rs_height, rs_width};
            runVar  = tensor<U>{rs_n_batch, rs_channels, rs_depth, rs_height, rs_width};

            const double Data_scale = 0.001;
            for(std::size_t i = 0; i < runMean.desc.GetElementSize(); i++)
            {
                runMean[i] = prng::gen_descreet_uniform_sign<U>(Data_scale, 100);
                runVar[i]  = prng::gen_descreet_unsigned<U>(Data_scale, 100);
            }
        }
        auto saveMean   = tensor<U>{rs_n_batch, rs_channels, rs_depth, rs_height, rs_width};
        auto saveInvVar = tensor<U>{rs_n_batch, rs_channels, rs_depth, rs_height, rs_width};
        auto out        = input;
        std::fill(out.begin(), out.end(), 0);

        const unsigned int in_cstride = depth * height * width;
        const auto ndhw               = double(in_cstride * n_batch);

        par_for(channels, 1, [&](int cidx) {
            double elemStd        = 0.;
            double variance_accum = 0.;
            double mean_accum     = 0.;
            double invVar         = 0.;
            double newRunMean     = 0.;
            double adjust         = 0.;

#if(MIO_HEIRARCH_SEL == 1)
            std::vector<double> variance_accum_arr(depth * height, 0.0);
            std::vector<double> mean_accum_arr(depth * height, 0.0);
            std::vector<double> dshift_accum_arr(depth * height, 0.0);
            std::vector<double> dscale_accum_arr(depth * height, 0.0);
#endif

#if(MIO_HEIRARCH_SEL == 0)
            // process the batch per channel
            for(std::size_t bidx = 0; bidx < n_batch; bidx++)
            { // via mini_batch
                for(std::size_t didx = 0; didx < depth; ++didx)
                { // via depth
                    for(std::size_t row = 0; row < height; row++)
                    { // via rows
                        for(std::size_t column = 0; column < width; column++)
                        { // via columns
                            // #1 calculate the mean
                            // iterating through the stack of images in the mini_batch
                            mean_accum += input(bidx, cidx, didx, row, column);
                        } // end for (column)
                    }     // end for (row)
                }         // end for (depth)
            }             // end for (n)
#else
            for(std::size_t didx = 0; didx < depth; ++didx)
            { // via depth
                for(std::size_t row = 0; row < height; row++){ //via rows
                    for(std::size_t column = 0; column < width; column++){// via columns
                        for(std::size_t bidx = 0; bidx < n_batch; bidx++){ //via mini_batch
                            mean_accum_arr[row] += input(bidx,cidx,didx, row,column);
                        }
                    }// for (column)
                }// for (row)
            } // for (depth)
            for(std::size_t i = 0; i<height; i++) mean_accum += mean_accum_arr[i];
#endif
            mean_accum /= ndhw;

            elemStd        = 0.;
            variance_accum = 0.;

#if(MIO_HEIRARCH_SEL == 0)
            // #2 calculate the variances
            // sigma^2 = (1/batch_mean) * sum( (x_i - batch_mean)^2 )
            for(std::size_t bidx = 0; bidx < n_batch; bidx++)
            { // via mini_batch
                for(std::size_t didx = 0; didx < depth; ++didx)
                { // via depth
                    for(std::size_t row = 0; row < height; row++)
                    { // via rows
                        for(std::size_t column = 0; column < width; column++)
                        { // via columns
                            // using out buffer as scratchpad
                            out(bidx, cidx, didx, row, column) = elemStd =
                                (input(bidx, cidx, didx, row, column) - mean_accum); // (x_i - mean)
                            variance_accum += (elemStd * elemStd); // sum{ (x_i - mean)^2 }
                        }                                          // end for (column)
                    }                                              // end for (row)
                }                                                  // for (depth)
            }                                                      // end for(n)

#else
            for(std::size_t didx = 0; didx < depth; ++didx) // via depth
            {
                for(std::size_t row = 0; row < height; row++){ //via rows
                    for(std::size_t column = 0; column < width; column++){// via columns
                        for(std::size_t bidx = 0; bidx < n_batch; bidx++){ //via mini_batch
                            out(bidx,cidx,didx,row,column) = elemStd = input(bidx,cidx,didx,row,column) - mean_accum;
                            variance_accum_arr[row] += elemStd*elemStd;
                        }
                    }// for (column)
                }// for (row)
            }
            for(std::size_t i = 0; i<height; i++) variance_accum += variance_accum_arr[i];
#endif
            variance_accum /= ndhw; // (1/N)*sum{ (x_i - mean)^2 }
            // #3 add epsilon for numeric stability, sqr_root, and invert
            invVar = 1.0 / sqrt(variance_accum + epsilon);

            // #4 apply the normalization
            // x_hat = (x_i - mean) / sqrt(variance_accum + epsilon)
            for(std::size_t bidx = 0; bidx < n_batch; bidx++)
            { // via mini_batch
                for(std::size_t didx = 0; didx < depth; ++didx)
                { // via depth
                    for(std::size_t row = 0; row < height; row++)
                    { // via rows
                        for(std::size_t column = 0; column < width; column++)
                        { // via columns
                            // #5 Gamma and Beta adjust
                            // y_i = gamma*x_hat + beta
                            out(bidx, cidx, didx, row, column) =
                                scale(0, cidx, 0, 0, 0) *
                                    (invVar * out(bidx, cidx, didx, row, column)) +
                                shift(0, cidx, 0, 0, 0);
                        } // for (column)
                    }     // for (row)
                }
            } // end for(n_batchs)

            saveMean(0, cidx, 0, 0, 0)   = mean_accum;
            saveInvVar(0, cidx, 0, 0, 0) = invVar;

            newRunMean = runMean(0, cidx, 0, 0, 0) * (1 - expAvgFactor);
            runMean(0, cidx, 0, 0, 0) =
                mean_accum * expAvgFactor + newRunMean; // newMean*factor + tmp
            // var(n+1) = p * var(n-1) + (1 - p)*(b/b-1)*var(n)
            adjust = (n_batch * depth * height * width == 1) ? variance_accum
                                                             : (ndhw / (ndhw - 1)) * variance_accum;
            runVar(0, cidx, 0, 0, 0) =
                (1 - expAvgFactor) * runVar(0, cidx, 0, 0, 0) + expAvgFactor * adjust;
        });

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward_train_3d_bn_spatial pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return std::make_tuple(out, runMean, runVar, saveMean, saveInvVar);
    }

    std::tuple<tensor<T>, tensor<U>, tensor<U>, tensor<U>, tensor<U>> gpu() const
    {

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        auto&& handle = get_handle();

        std::size_t n_batch, channels, depth, height, width;
        std::tie(n_batch, channels, depth, height, width) =
            miopen::tien<5>(input.desc.GetLengths());

        auto out = input;
        std::fill(out.begin(), out.end(), 0);

        std::size_t rs_n_batch, rs_channels, rs_depth, rs_height, rs_width;
        auto derivedBnDesc = miopen::TensorDescriptor{};

        miopen::DeriveBNTensorDescriptor(derivedBnDesc, input.desc, miopenBNSpatial);

        std::tie(rs_n_batch, rs_channels, rs_depth, rs_height, rs_width) =
            miopen::tien<5>(derivedBnDesc.GetLengths());

        tensor<U> runMean;
        tensor<U> runVar;

        if(input.desc.GetType() == miopenFloat)
        {
            runMean = tensor<U>{rs_n_batch, rs_channels, rs_depth, rs_height, rs_width}.generate(
                tensor_elem_gen_integer{17});
            runVar = tensor<U>{rs_n_batch, rs_channels, rs_depth, rs_height, rs_width}.generate(
                tensor_elem_gen_integer{17});
        }
        else
        {
            prng::reset_seed();
            runMean = tensor<U>{rs_n_batch, rs_channels, rs_depth, rs_height, rs_width};
            runVar  = tensor<U>{rs_n_batch, rs_channels, rs_depth, rs_height, rs_width};

            const double Data_scale = 0.001;
            for(std::size_t i = 0; i < runMean.desc.GetElementSize(); i++)
            {
                runMean[i] = prng::gen_descreet_uniform_sign<U>(Data_scale, 100);
                runVar[i]  = prng::gen_descreet_unsigned<U>(Data_scale, 100);
            }
        }

        auto saveMean   = tensor<U>{rs_n_batch, rs_channels, rs_depth, rs_height, rs_width};
        auto saveInvVar = tensor<U>{rs_n_batch, rs_channels, rs_depth, rs_height, rs_width};

        // in buffers
        auto in_dev    = handle.Write(input.data);
        auto scale_dev = handle.Write(scale.data);
        auto shift_dev = handle.Write(shift.data);

        // out buffers
        auto runMean_dev    = handle.Write(runMean.data);
        auto runVar_dev     = handle.Write(runVar.data);
        auto saveMean_dev   = handle.Create<U>(channels);
        auto saveInvVar_dev = handle.Create<U>(channels);
        auto out_dev        = handle.Create<T>(n_batch * channels * depth * height * width);

        double epsilon      = MIO_BN_TEST_EPSILON;
        double expAvgFactor = MIO_BN_TEST_EXPAVGFACTOR;

        float alpha = 1.0;
        float beta  = 0.0;

        miopen::BatchNormForwardTraining(handle,
                                         miopenBNSpatial,
                                         &alpha,
                                         &beta,
                                         miopen::BuildReshaped4DTensorDescriptor(input.desc),
                                         in_dev.get(),
                                         miopen::BuildReshaped4DTensorDescriptor(out.desc),
                                         out_dev.get(),
                                         miopen::BuildReshaped4DTensorDescriptor(scale.desc),
                                         miopen::BuildReshaped4DTensorDescriptor(shift.desc),
                                         miopen::BuildReshaped4DTensorDescriptor(shift.desc),
                                         miopen::BuildReshaped4DTensorDescriptor(shift.desc),
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

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU forward_train_3d_bn_spatial pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif

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

//****************************************************
// FORWARD INFERENCE
//****************************************************
template <class T, class U>
struct verify_forward_infer_3d_bn_spatial_recalc
{

    const tensor<T> input;
    const tensor<U> scale;
    const tensor<U> shift;

    tensor<T> cpu() const
    {

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        double epsilon = MIO_BN_TEST_EPSILON;

        std::size_t n_batch, channels, depth, height, width;
        std::tie(n_batch, channels, depth, height, width) =
            miopen::tien<5>(input.desc.GetLengths());

        auto out = input;
        std::fill(out.begin(), out.end(), 0);

        const unsigned int in_cstride = depth * height * width;
        const auto ndhw               = double(in_cstride * n_batch);

        par_for(channels, 1, [&](int cidx) {
            double elemStd        = 0.;
            double variance_accum = 0.;
            double mean_accum     = 0.;
            double inhat          = 0.;
            double invVar         = 0.;

            // process the batch per channel
            for(std::size_t didx = 0; didx < depth; ++didx)
            { // via rows
                for(std::size_t row = 0; row < height; row++)
                { // via rows
                    for(std::size_t column = 0; column < width; column++)
                    { // via columns
                        // #1 calculate the mean
                        for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                        { // via mini_batch
                            // iterating through the stack of images in the mini_batch
                            mean_accum += input(bidx, cidx, didx, row, column);
                        } // end for (n)
                    }     // end for (column)
                }         // end for (row)
            }
            mean_accum /= ndhw;

            elemStd        = 0.;
            variance_accum = 0.;
            // #2 calculate the variances
            // sigma^2 = (1/batch_mean) * sum( (x_i - batch_mean)^2 )
            for(std::size_t didx = 0; didx < depth; ++didx)
            { // via depth
                for(std::size_t row = 0; row < height; row++)
                { // via rows
                    for(std::size_t column = 0; column < width; column++)
                    { // via columns
                        for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                        { // via mini_batch
                            // using out buffer as scratchpad
                            out(bidx, cidx, didx, row, column) = elemStd =
                                (input(bidx, cidx, didx, row, column) - mean_accum); // (x_i - mean)
                            variance_accum += (elemStd * elemStd); // sum{ (x_i - mean)^2 }
                        }                                          // end for(n)
                    }                                              // end for (column)
                }                                                  // end for (row)
            }                                                      // end for (depth)
            variance_accum /= ndhw;                                // (1/N)*sum{ (x_i - mean)^2 }

            // #3 add epsilon for numeric stability, sqr_root, and invert
            invVar = 1.0 / sqrt(variance_accum + epsilon);

            // #4 apply the normalization
            // x_hat = (x_i - mean) / sqrt(variance_accum - epsilon)
            for(std::size_t didx = 0; didx < depth; ++didx)
            { // via depth
                for(std::size_t row = 0; row < height; row++)
                { // via rows
                    for(std::size_t column = 0; column < width; column++)
                    { // via columns
                        for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                        { // via mini_batch
                            elemStd = out(bidx,
                                          cidx,
                                          didx,
                                          row,
                                          column); // using saved values from output tensor
                            inhat   = elemStd * invVar;
                            // #5 Gamma and Beta adjust // y_i = gamma*x_hat + beta
                            out(bidx, cidx, didx, row, column) =
                                scale(0, cidx, 0, 0, 0) * inhat + shift(0, cidx, 0, 0, 0);
                        } // end for(n_batchs)
                    }     // for (column)
                }         // for (row)
            }             // for (depth)
        });

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward_infer_3d_bn_spatial_recalc pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return out;
    }

    tensor<T> gpu() const
    {

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        auto&& handle = get_handle();
        auto out      = input;
        std::fill(out.begin(), out.end(), 0);

        auto in_dev    = handle.Write(input.data);
        auto scale_dev = handle.Write(scale.data);
        auto shift_dev = handle.Write(shift.data);
        auto out_dev   = handle.Write(out.data);

        float alpha = 1.0;
        float beta  = 0.0;

        double epsilon = MIO_BN_TEST_EPSILON;

        miopen::BatchNormForwardInference(handle,
                                          miopenBNSpatial,
                                          &alpha,
                                          &beta,
                                          miopen::BuildReshaped4DTensorDescriptor(input.desc),
                                          in_dev.get(),
                                          miopen::BuildReshaped4DTensorDescriptor(out.desc),
                                          out_dev.get(),
                                          miopen::BuildReshaped4DTensorDescriptor(scale.desc),
                                          miopen::BuildReshaped4DTensorDescriptor(shift.desc),
                                          miopen::BuildReshaped4DTensorDescriptor(shift.desc),
                                          miopen::BuildReshaped4DTensorDescriptor(shift.desc),
                                          scale_dev.get(),
                                          shift_dev.get(),
                                          nullptr,
                                          nullptr,
                                          epsilon);
        out.data = handle.Read<T>(out_dev, out.data.size());

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU forward_infer_3d_bn_spatial_recalc pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return out;
    }

    void fail(int) const
    {
        std::cout << "Forward Inference Spatial Batch Normalization Recalc: " << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
    }
};

template <class T, class U>
struct verify_forward_infer_3d_bn_spatial_use_est
{

    const tensor<T> input;
    const tensor<U> scale;
    const tensor<U> shift;
    const tensor<U> estMean;
    const tensor<U> estVar;
    tensor<T> cpu() const
    {

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        double epsilon = MIO_BN_TEST_EPSILON;

        std::size_t n_batch, channels, depth, height, width;
        std::tie(n_batch, channels, depth, height, width) =
            miopen::tien<5>(input.desc.GetLengths());

        auto out = input;
        std::fill(out.begin(), out.end(), 0);

        par_for(channels, 1, [&](int cidx) {
            double elemStd  = 0.;
            double variance = estVar(0, cidx, 0, 0, 0);
            double mean     = estMean(0, cidx, 0, 0, 0);
            double inhat    = 0.;
            double invVar   = 1.0 / sqrt(variance + epsilon);

            // process the batch per channel
            for(std::size_t bidx = 0; bidx < n_batch; bidx++)
            { // via mini_batch
                for(std::size_t didx = 0; didx < depth; ++didx)
                { // via depth
                    for(std::size_t row = 0; row < height; row++)
                    { // via rows
                        for(std::size_t column = 0; column < width; column++)
                        { // via columns

                            elemStd = input(bidx, cidx, didx, row, column) - mean;
                            inhat   = elemStd * invVar;
                            out(bidx, cidx, didx, row, column) =
                                scale(0, cidx, 0, 0, 0) * inhat + shift(0, cidx, 0, 0, 0);
                        }
                    }
                }
            }
        });
#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU forward_infer_3d_bn_spatial_use_est pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif

        return out;
    }

    tensor<T> gpu() const
    {
#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        auto&& handle = get_handle();
        auto out      = input;
        std::fill(out.begin(), out.end(), 0);

        auto in_dev      = handle.Write(input.data);
        auto estMean_dev = handle.Write(estMean.data);
        auto estVar_dev  = handle.Write(estVar.data);
        auto scale_dev   = handle.Write(scale.data);
        auto shift_dev   = handle.Write(shift.data);
        auto out_dev     = handle.Write(out.data);

        float alpha = 1.0;
        float beta  = 0.0;

        double epsilon = MIO_BN_TEST_EPSILON;

        miopen::BatchNormForwardInference(handle,
                                          miopenBNSpatial,
                                          &alpha,
                                          &beta,
                                          miopen::BuildReshaped4DTensorDescriptor(input.desc),
                                          in_dev.get(),
                                          miopen::BuildReshaped4DTensorDescriptor(out.desc),
                                          out_dev.get(),
                                          miopen::BuildReshaped4DTensorDescriptor(scale.desc),
                                          miopen::BuildReshaped4DTensorDescriptor(shift.desc),
                                          miopen::BuildReshaped4DTensorDescriptor(shift.desc),
                                          miopen::BuildReshaped4DTensorDescriptor(shift.desc),
                                          scale_dev.get(),
                                          shift_dev.get(),
                                          estMean_dev.get(),
                                          estVar_dev.get(),
                                          epsilon);
        out.data = handle.Read<T>(out_dev, out.data.size());
#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU forward_infer_3d_bn_spatial_use_est pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return out;
    }

    void fail(int) const
    {
        std::cout << "Forward Inference Spatial Batch Normalization Use Estimated: " << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
    }
};

//****************************************************
// BACKWARDS PROPAGATION
//****************************************************
template <class T, class U>
struct verify_backward_3d_bn_spatial_recalc
{

    const tensor<T> x_input;
    const tensor<T> dy_input;
    const tensor<U> scale;

    std::tuple<tensor<T>, tensor<U>, tensor<U>> cpu() const
    {

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        double epsilon = MIO_BN_TEST_EPSILON;

        std::size_t n_batch, channels, depth, height, width;
        std::tie(n_batch, channels, depth, height, width) =
            miopen::tien<5>(x_input.desc.GetLengths());

        std::size_t ss_n_batch, ss_channels, ss_depth, ss_height, ss_width;
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, x_input.desc, miopenBNSpatial);
        std::tie(ss_n_batch, ss_channels, ss_depth, ss_height, ss_width) =
            miopen::tien<5>(derivedBnDesc.GetLengths());

        auto dx_out = tensor<T>{n_batch, channels, depth, height, width};
        std::fill(dx_out.begin(), dx_out.end(), 0);

        auto dscale = tensor<U>{ss_n_batch, ss_channels, ss_depth, ss_height, ss_width};
        std::fill(dscale.begin(), dscale.end(), 0);

        auto dshift = tensor<U>{ss_n_batch, ss_channels, ss_depth, ss_height, ss_width};
        std::fill(dshift.begin(), dshift.end(), 0);

        const unsigned int in_dstride = height * width;
        const unsigned int in_cstride = depth * in_dstride;
        const auto ndhw               = double(in_cstride * n_batch);

        par_for(channels, 1, [&](int cidx) {
            double elemStd = 0.;
            unsigned int xhat_index;
            double mean     = 0.;
            double invVar   = 0.;
            double dyelem   = 0.;
            double variance = 0.;

            std::vector<double> xhat(n_batch * in_cstride, 0.0);

#if(MIO_HEIRARCH_SEL == 1)
            std::vector<double> variance_accum_arr(depth * height, 0.0);
            std::vector<double> mean_accum_arr(depth * height, 0.0);
            std::vector<double> dshift_accum_arr(depth * height, 0.0);
            std::vector<double> dscale_accum_arr(depth * height, 0.0);
#endif

// process the batch per channel
#if(MIO_HEIRARCH_SEL == 0)
            for(std::size_t didx = 0; didx < depth; ++didx)
            { // via depth
                for(std::size_t row = 0; row < height; row++)
                { // via rows
                    for(std::size_t column = 0; column < width; column++)
                    { // via columns
                        for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                        { // via mini_batch
                            // #1 calculate the mean
                            mean += x_input(bidx, cidx, didx, row, column);
                        }
                    } // for (column)
                }     // for (row)
            }         // for (depth)
#else
            for(std::size_t didx = 0; didx < depth; ++didx)
            { // via depth
                for(std::size_t row = 0; row < height; row++){ //via rows
                    for(std::size_t column = 0; column < width; column++){// via columns
                        for(std::size_t bidx = 0; bidx < n_batch; bidx++){ //via mini_batch
                            mean_accum_arr[row] += x_input(bidx,cidx,didx,row,column);
                        }
                    }// for (column)
                }// for (row)
            }
            for(std::size_t i = 0; i<height; i++) mean += mean_accum_arr[i];
#endif
            mean /= ndhw;

            elemStd  = 0.;
            variance = 0.;
#if(MIO_HEIRARCH_SEL == 0)
            for(std::size_t didx = 0; didx < depth; ++didx)
            { // via depth
                for(std::size_t row = 0; row < height; row++)
                { // via rows
                    for(std::size_t column = 0; column < width; column++)
                    { // via columns
                        // #2 calculate the variances
                        // sigma^2 = (1/batch_mean) * sum( (x_i - batch_mean)^2 )
                        for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                        { // via mini_batch
                            // per (x-dims) channel load a block of data into LDS
                            elemStd = x_input(bidx, cidx, didx, row, column) - mean; // (x_i - mean)
                            variance += elemStd * elemStd; // sum{ (x_i - mean)^2 }
                        }                                  // end for(n)
                    }                                      // for (column)
                }                                          // for (row)
            }
#else
            for(std::size_t didx = 0; didx < depth; ++didx)
            { //
                for(std::size_t row = 0; row < height; row++){ //via rows
                    for(std::size_t column = 0; column < width; column++){// via columns
                        for(std::size_t bidx = 0; bidx < n_batch; bidx++){ //via mini_batch
                            elemStd = x_input(bidx,cidx,didx,row,column) - mean;
                            variance_accum_arr[row] += elemStd*elemStd;
                        }
                    }// for (column)
                }// for (row)
            }
            for(std::size_t i = 0; i<height; i++) variance += variance_accum_arr[i];
#endif
            variance /= ndhw; // (1/(N*H*W))*sum{ (x_i - mean)^2 }
            invVar = 1. / double(sqrt(variance + epsilon));

            dscale(0, cidx, 0, 0, 0) = 0.;

#if(MIO_HEIRARCH_SEL == 0)
            for(std::size_t didx = 0; didx < depth; ++didx)
            { //
                for(std::size_t row = 0; row < height; row++)
                { // via rows
                    for(std::size_t column = 0; column < width; column++)
                    { // via columns
                        for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                        { // via mini_batch
                            xhat_index =
                                in_cstride * bidx + in_dstride * didx + width * row + column;
                            // per (x-dims) channel load a block of data into LDS
                            elemStd = x_input(bidx, cidx, didx, row, column) - mean; // (x_i - mean)
                            xhat[xhat_index] = elemStd * invVar;
                            dyelem           = dy_input(bidx, cidx, didx, row, column);
                            dshift(0, cidx, 0, 0, 0) += dyelem;
                            dscale(0, cidx, 0, 0, 0) += xhat[xhat_index] * dyelem;
                        } // end for(n_batch)
                    }     // for (column)
                }         // for (row)
            }
#else
            for(std::size_t didx = 0; didx < depth; ++didx)
            {
                for(std::size_t row = 0; row < height; row++){ //via rows
                    for(std::size_t column = 0; column < width; column++){// via columns
                        for(std::size_t bidx = 0; bidx < n_batch; bidx++){ //via mini_batch
                            xhat_index = in_cstride*bidx + in_dstride*didx + width*row + column;
                            //per (x-dims) channel load a block of data into LDS
                            elemStd             = x_input(bidx,cidx,didx,row,column) - mean;// (x_i - mean)
                            xhat[xhat_index]    = elemStd*invVar;
                            dyelem              = dy_input(bidx,cidx,didx,row,column);
                            dshift_accum_arr[row] += dyelem;
                            dscale_accum_arr[row] += xhat[xhat_index]*dyelem;
                            //dscale_accum_arr[row] += x_input(bidx,cidx,row,column);;//dscale_accum_arr[row] += xhat[xhat_index];
                            //dscale_accum_arr[row] += 1.0;//DEBUG
                        }
                    }// for (column)
                }// for (row)
            }
            for(std::size_t i = 0; i<height; i++) {
                dshift(0,cidx,0,0,0) += dshift_accum_arr[i];
                dscale(0,cidx,0,0,0) += dscale_accum_arr[i];
            }
#endif
            for(std::size_t didx = 0; didx < depth; ++didx)
            { //
                for(std::size_t row = 0; row < height; row++)
                { // via rows
                    for(std::size_t column = 0; column < width; column++)
                    { // via columns
                        for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                        { // via mini_batch
                            xhat_index =
                                in_cstride * bidx + in_dstride * didx + width * row + column;

                            double tmp1 = ndhw * dy_input(bidx, cidx, didx, row, column) -
                                          dshift(0, cidx, 0, 0, 0);
                            double tmp2 = -xhat[xhat_index] * dscale(0, cidx, 0, 0, 0);
                            double tmp3 = (scale(0, cidx, 0, 0, 0) * invVar) / ndhw;
                            dx_out(bidx, cidx, didx, row, column) = tmp3 * (tmp2 + tmp1);
                        } // end for(n_batchs)
                    }     // for (column)
                }         // for (row)
            }
        }); // for (channel)

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU backward_3d_bn_spatial_recalc pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif

        return std::make_tuple(dx_out, dscale, dshift);
    }

    std::tuple<tensor<T>, tensor<U>, tensor<U>> gpu() const
    {
#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        auto&& handle = get_handle();

        std::size_t n_batch, channels, depth, height, width;
        std::tie(n_batch, channels, depth, height, width) =
            miopen::tien<5>(x_input.desc.GetLengths());

        auto dx_out = tensor<T>{n_batch, channels, depth, height, width};
        std::fill(dx_out.begin(), dx_out.end(), 0);

        std::size_t ss_n_batch, ss_channels, ss_depth, ss_height, ss_width;
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, x_input.desc, miopenBNSpatial);
        std::tie(ss_n_batch, ss_channels, ss_depth, ss_height, ss_width) =
            miopen::tien<5>(derivedBnDesc.GetLengths());

        auto dscale = tensor<U>{ss_n_batch, ss_channels, ss_depth, ss_height, ss_width};
        std::fill(dscale.begin(), dscale.end(), 0);

        auto dshift = tensor<U>{ss_n_batch, ss_channels, ss_depth, ss_height, ss_width};
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
                                  miopen::BuildReshaped4DTensorDescriptor(x_input.desc),
                                  xin_dev.get(),
                                  miopen::BuildReshaped4DTensorDescriptor(dy_input.desc),
                                  dyin_dev.get(),
                                  miopen::BuildReshaped4DTensorDescriptor(dx_out.desc),
                                  dx_out_dev.get(),
                                  miopen::BuildReshaped4DTensorDescriptor(scale.desc),
                                  miopen::BuildReshaped4DTensorDescriptor(dshift.desc),
                                  miopen::BuildReshaped4DTensorDescriptor(dshift.desc),
                                  miopen::BuildReshaped4DTensorDescriptor(dshift.desc),
                                  scale_dev.get(),
                                  dscale_dev.get(),
                                  dshift_dev.get(),
                                  epsilon,
                                  nullptr,
                                  nullptr);

        dx_out.data = handle.Read<T>(dx_out_dev, dx_out.data.size());
        dscale.data = handle.Read<U>(dscale_dev, dscale.data.size());
        dshift.data = handle.Read<U>(dshift_dev, dshift.data.size());

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU backward_3d_bn_spatial_recalc pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
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
struct verify_backward_3d_bn_spatial_use_saved
{

    const tensor<T> x_input;
    const tensor<T> dy_input;
    const tensor<U> scale;
    const tensor<U> savedMean;
    const tensor<U> savedInvVar;
    std::tuple<tensor<T>, tensor<U>, tensor<U>> cpu() const
    {

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif

        std::size_t n_batch, channels, depth, height, width;
        std::tie(n_batch, channels, depth, height, width) =
            miopen::tien<5>(x_input.desc.GetLengths());

        auto dx_out = tensor<T>{n_batch, channels, depth, height, width};
        std::fill(dx_out.begin(), dx_out.end(), 0);

        std::size_t ss_n_batch, ss_channels, ss_depth, ss_height, ss_width;
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, x_input.desc, miopenBNSpatial);
        std::tie(ss_n_batch, ss_channels, ss_depth, ss_height, ss_width) =
            miopen::tien<5>(derivedBnDesc.GetLengths());

        auto dscale = tensor<U>{ss_n_batch, ss_channels, ss_depth, ss_height, ss_width};
        std::fill(dscale.begin(), dscale.end(), 0);

        auto dshift = tensor<U>{ss_n_batch, ss_channels, ss_depth, ss_height, ss_width};
        std::fill(dshift.begin(), dshift.end(), 0);

        const unsigned int in_dstride = height * width;
        const unsigned int in_cstride = depth * in_dstride;
        const auto ndhw               = double(in_cstride * n_batch);

        par_for(channels, 1, [&](int cidx) {
            double elemStd = 0.;
            unsigned int xhat_index;
            double mean   = savedMean(0, cidx, 0, 0, 0);   // HxW elements
            double invVar = savedInvVar(0, cidx, 0, 0, 0); // HxW elements
            double dyelem = 0.;

            std::vector<double> xhat(n_batch * in_cstride, 0.0);

#if(MIO_HEIRARCH_SEL == 1)
            std::vector<double> dshift_accum_arr(height * depth, 0.0);
            std::vector<double> dscale_accum_arr(height * depth, 0.0);
#endif

            // process the batch per channel
            dscale(0, cidx, 0, 0, 0) = 0.;

#if(MIO_HEIRARCH_SEL == 0)
            for(std::size_t didx = 0; didx < depth; ++didx)
            { // via depth
                for(std::size_t row = 0; row < height; row++)
                { // via rows
                    for(std::size_t column = 0; column < width; column++)
                    { // via columns
                        for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                        { // via mini_batch
                            xhat_index =
                                in_cstride * bidx + in_dstride * didx + width * row + column;
                            // per (x-dims) channel load a block of data into LDS
                            elemStd = x_input(bidx, cidx, didx, row, column) - mean; // (x_i - mean)
                            xhat[xhat_index] = elemStd * invVar;
                            dyelem           = dy_input(bidx, cidx, didx, row, column);
                            dshift(0, cidx, 0, 0, 0) += dyelem;
                            dscale(0, cidx, 0, 0, 0) += xhat[xhat_index] * dyelem;
                        } // end for(n_batch)
                    }     // for (column)
                }         // for (row)
            }
#else
            for(std::size_t didx = 0; didx < depth; ++didx)
            { // via depth
                for(std::size_t row = 0; row < height; row++){ //via rows
                    for(std::size_t column = 0; column < width; column++){// via columns
                        for(std::size_t bidx = 0; bidx < n_batch; bidx++){ //via mini_batch
                            xhat_index = in_cstride*bidx + in_dstride*didx + width*row + column;
                            //per (x-dims) channel load a block of data into LDS
                            elemStd             = x_input(bidx,cidx,didx,row,column) - mean;// (x_i - mean)
                            xhat[xhat_index]    = elemStd*invVar;
                            //printf("xhat[%d]: %lf\n",xhat_index,xhat[xhat_index]);
                            dyelem              = dy_input(bidx,cidx,didx, row,column);
                            dshift_accum_arr[row] += dyelem;
                            dscale_accum_arr[row] += xhat[xhat_index]*dyelem;
                            //dscale_accum_arr[row] += 1.0;//DEBUG
                        }
                    }// for (column)
                }// for (row)
            }

            for(std::size_t i = 0; i<height; i++) {
                dshift(0,cidx,0,0,0) += dshift_accum_arr[i];
                dscale(0,cidx,0,0,0) += dscale_accum_arr[i];
            }
#endif
            for(std::size_t didx = 0; didx < depth; ++didx)
            { // via depth
                for(std::size_t row = 0; row < height; row++)
                { // via rows
                    for(std::size_t column = 0; column < width; column++)
                    { // via columns
                        for(std::size_t bidx = 0; bidx < n_batch; bidx++)
                        { // via mini_batch
                            xhat_index =
                                in_cstride * bidx + in_dstride * didx + width * row + column;

                            double tmp1 = ndhw * dy_input(bidx, cidx, didx, row, column) -
                                          dshift(0, cidx, 0, 0, 0);
                            double tmp2 = -xhat[xhat_index] * dscale(0, cidx, 0, 0, 0);
                            double tmp3 = (scale(0, cidx, 0, 0, 0) * invVar) / ndhw;
                            dx_out(bidx, cidx, didx, row, column) = tmp3 * (tmp2 + tmp1);
                        } // end for(n_batchs)
                    }     // for (column)
                }         // for (row)
            }
        }); // for (channel)
#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: CPU backward_bn spatial_use_saved pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif
        return std::make_tuple(dx_out, dscale, dshift);
    }

    std::tuple<tensor<T>, tensor<U>, tensor<U>> gpu() const
    {

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_start = std::chrono::high_resolution_clock::now();
#endif
        auto&& handle = get_handle();

        std::size_t n_batch, channels, depth, height, width;
        std::tie(n_batch, channels, depth, height, width) =
            miopen::tien<5>(x_input.desc.GetLengths());

        auto dx_out = tensor<T>{n_batch, channels, depth, height, width};
        std::fill(dx_out.begin(), dx_out.end(), 0);

        std::size_t ss_n_batch, ss_channels, ss_depth, ss_height, ss_width;
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, x_input.desc, miopenBNSpatial);
        std::tie(ss_n_batch, ss_channels, ss_depth, ss_height, ss_width) =
            miopen::tien<5>(derivedBnDesc.GetLengths());

        auto dscale = tensor<U>{ss_n_batch, ss_channels, ss_depth, ss_height, ss_width};
        std::fill(dscale.begin(), dscale.end(), 0);

        auto dshift = tensor<U>{ss_n_batch, ss_channels, ss_depth, ss_height, ss_width};
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
                                  miopen::BuildReshaped4DTensorDescriptor(x_input.desc),
                                  xin_dev.get(),
                                  miopen::BuildReshaped4DTensorDescriptor(dy_input.desc),
                                  dyin_dev.get(),
                                  miopen::BuildReshaped4DTensorDescriptor(dx_out.desc),
                                  dx_out_dev.get(),
                                  miopen::BuildReshaped4DTensorDescriptor(scale.desc),
                                  miopen::BuildReshaped4DTensorDescriptor(dshift.desc),
                                  miopen::BuildReshaped4DTensorDescriptor(dshift.desc),
                                  miopen::BuildReshaped4DTensorDescriptor(dshift.desc),
                                  scale_dev.get(),
                                  dscale_dev.get(),
                                  dshift_dev.get(),
                                  epsilon,
                                  savedMean_dev.get(),
                                  savedInvVar_dev.get());

        dx_out.data = handle.Read<T>(dx_out_dev, dx_out.data.size());
        dscale.data = handle.Read<U>(dscale_dev, dscale.data.size());
        dshift.data = handle.Read<U>(dshift_dev, dshift.data.size());

#if(MIO_BN_TIME_EVERYTHING == 1)
        auto t_end = std::chrono::high_resolution_clock::now();

        std::cout << "Wall clock: GPU backward_3d_bn_spatial_use_saved pass time: "
                  << std::chrono::duration<double>(t_end - t_start).count() << " seconds."
                  << std::endl;
#endif

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

//====== DRIVERS ===========================================
template <class T>
struct batch_norm_3d_spatial_driver : test_driver
{
    tensor<T> input;
    tensor<PREC_TYPE> scale;
    tensor<PREC_TYPE> shift;
    batch_norm_3d_spatial_driver()
    {
        this->batch_factor = 4;
        this->tolerance =
            4e-3 / std::numeric_limits<T>::epsilon(); // ck solver has tolerance of 4e-3
        add(input,
            "input",
            get_3d_bn_spatial_input_tensor(
                tensor_elem_gen_integer{miopen_type<T>{} == miopenHalf ? 5 : 17}));
    }

    void run()
    {
        std::size_t n, c, d, h, w;
        std::tie(n, c, d, h, w) = miopen::tien<5>(input.desc.GetLengths());

        // The condition is derived form bn_spatial_test.cpp as they are known failures
        if(n == 1)
        {
            std::cout << "(n=1) is not supported for BN operation." << std::endl;
            return;
        }

        if((h * w * d > 1024) && (input.desc.GetType() == miopenHalf) && (MIO_BN_USE_MIX_PREC == 0))
        {
            std::cout << "(h*w*d > 1024) is not supported for BN operations "
                      << "when half precision is used but mixed precision disabled." << std::endl;
            return;
        }

        std::size_t ssn, ssc, ssd, ssh, ssw;
        auto derivedBnDesc = miopen::TensorDescriptor{};
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, input.desc, miopenBNSpatial);
        std::tie(ssn, ssc, ssd, ssh, ssw) = miopen::tien<5>(derivedBnDesc.GetLengths());

        scale                   = tensor<PREC_TYPE>{ssn, ssc, ssd, ssh, ssw};
        shift                   = tensor<PREC_TYPE>{ssn, ssc, ssd, ssh, ssw};
        const double Data_scale = 1e-4;

        for(std::size_t i = 0; i < scale.desc.GetElementSize(); i++)
        {
            scale[i] = prng::gen_descreet_uniform_sign<PREC_TYPE>(Data_scale, 100);
            shift[i] = prng::gen_descreet_uniform_sign<PREC_TYPE>(Data_scale, 100);
        }
        for(std::size_t i = 0; i < input.desc.GetElementSize(); i++)
        {
            input[i] = prng::gen_descreet_uniform_sign<T>(1e-5, 100);
        }

// train
#if(MIO_BN_SP_TEST_DEBUG == 1)
        std::cout << "Running forward train spatial with R and S set." << std::endl;
#endif
        auto outpair =
            verify(verify_forward_train_3d_bn_spatial<T, PREC_TYPE>{input, scale, shift});
// returns:  std::make_tuple(out,runMean,runVar,saveMean,saveInvVar);

// inference recalc
#if(MIO_BN_SP_TEST_DEBUG == 1)
        std::cout << "Running forward inference spatial recalc." << std::endl;
#endif
        // this->tolerance = 80;
        // Debug values
        // std::fill(input.begin(), input.end(), 1);
        // std::fill(scale.begin(), scale.end(), 1);
        // std::fill(shift.begin(), shift.end(), 1);
        verify(verify_forward_infer_3d_bn_spatial_recalc<T, PREC_TYPE>{input, scale, shift});

        // inference use estimated running values
        auto estMean = std::get<1>(outpair.second);
        auto estVar  = std::get<2>(outpair.second);
#if(MIO_BN_SP_TEST_DEBUG == 1)
        std::cout << "Running forward inference spatial with R set." << std::endl;
#endif
        verify(verify_forward_infer_3d_bn_spatial_use_est<T, PREC_TYPE>{
            input, scale, shift, estMean, estVar});

        // backprop recalc
        auto dy_input = std::get<0>(outpair.second);
        for(std::size_t bidx = 0; bidx < n; bidx++)
        { // via mini_batch
            for(std::size_t cidx = 0; cidx < c; cidx++)
            { // via mini_batch
                for(std::size_t didx = 0; didx < d; didx++)
                { // via depth
                    for(std::size_t row = 0; row < h; row++)
                    { // via rows
                        for(std::size_t column = 0; column < w; column++)
                        {
                            dy_input(bidx, cidx, didx, row, column) *= 0.1;
                        }
                    }
                }
            }
        }
#if(MIO_BN_SP_TEST_DEBUG == 2)
        auto debugvals = verify(verify_backward_3d_bn_spatial_recalc<T>{input, dy_input, scale});
        auto gpuout    = std::get<0>(debugvals.second);
        auto cpuout    = std::get<0>(debugvals.first);

        double maxdiff = 0.;
        int mn         = 0;
        int mc         = 0;
        int md         = 0;
        int mh         = 0;
        int mw         = 0;

        for(std::size_t bidx = 0; bidx < n; bidx++)
        { // via mini_batch
            for(std::size_t cidx = 0; cidx < c; cidx++)
            { // via mini_batch
                for(std::size_t didx = 0; didx < d; didx++)
                {
                    for(std::size_t row = 0; row < h; row++)
                    { // via rows
                        for(std::size_t column = 0; column < w; column++)
                        { // via columns
                            double diff = fabs(gpuout(bidx, cidx, didx, row, column) -
                                               cpuout(bidx, cidx, didx, row, column));
                            if(diff > maxdiff)
                            {
                                maxdiff = diff;
                                mn      = bidx;
                                mc      = cidx;
                                md      = didx;
                                mh      = row;
                                mw      = column;
                            }
                            // if(diff > 1.)
                            // {
                            std::cout << "gpu[" << bidx << ", " << cidx << ", " << didx << ", "
                                      << row << ", " << column
                                      << "]: " << gpuout(bidx, cidx, didx, row, column) << " :: ";
                            std::cout << "cpu[" << bidx << ", " << cidx << ", " << didx << ", "
                                      << row << ", " << column
                                      << "]: " << cpuout(bidx, cidx, didx, row, column) << " :: ";
                            std::cout << "diff: " << diff << std::endl;
                            //    }
                        }
                    }
                }
            }
        }
        if(maxdiff > 0)
        {
            std::cout << "Max diff: " << maxdiff << std::endl;
            std::cout << "gpu[" << mn << ", " << mc << ", " << md << ", " << mh << ", " << mw
                      << "]: " << gpuout(mn, mc, md, mh, mw) << " :: ";
            std::cout << "cpu[" << mn << ", " << mc << ", " << md << ", " << mh << ", " << mw
                      << "]: " << cpuout(mn, mc, md, mh, mw) << std::endl;
        }
#else
#if(MIO_BN_SP_TEST_DEBUG == 1)
        std::cout << "Running back propagation spatial recalc." << std::endl;
#endif
        this->tolerance = 80 * input.desc.GetElementSize();
        verify(verify_backward_3d_bn_spatial_recalc<T, PREC_TYPE>{input, dy_input, scale});
#endif

        // backprop use saved values
        auto savedMean   = std::get<3>(outpair.second);
        auto savedInvVar = std::get<4>(outpair.second);

#if(MIO_BN_SP_TEST_DEBUG == 3)

        auto debugvals = verify(verify_backward_3d_bn_spatial_use_saved<T>{
            input, dy_input, scale, savedMean, savedInvVar});
        auto gpuout    = std::get<0>(debugvals.second);
        auto cpuout    = std::get<0>(debugvals.first);

        double maxdiff = 0.;
        int mn         = 0;
        int mc         = 0;
        int md         = 0;
        int mh         = 0;
        int mw         = 0;

        for(std::size_t bidx = 0; bidx < n; bidx++)
        { // via mini_batch
            for(std::size_t cidx = 0; cidx < c; cidx++)
            { // via mini_batch
                for(std::size_t didx = 0; didx < d; didx++)
                { // via mini_batch
                    for(std::size_t row = 0; row < h; row++)
                    { // via rows
                        for(std::size_t column = 0; column < w; column++)
                        { // via columns
                            double diff = fabs(gpuout(bidx, cidx, didx, row, column) -
                                               cpuout(bidx, cidx, didx, row, column));
                            if(diff > maxdiff)
                            {
                                maxdiff = diff;
                                mn      = bidx;
                                mc      = cidx;
                                md      = didx;
                                mh      = row;
                                mw      = column;
                            }
                            // if(diff > 1.)
                            //{
                            std::cout << "gpu[" << bidx << ", " << cidx << ", " << didx << ", "
                                      << row << ", " << column
                                      << "]: " << gpuout(bidx, cidx, didx, row, column) << " :: ";
                            std::cout << "cpu[" << bidx << ", " << cidx << ", " << didx << ", "
                                      << row << ", " << column
                                      << "]: " << cpuout(bidx, cidx, didx, row, column) << " :: ";
                            std::cout << "diff: " << diff << std::endl;
                            //}
                        }
                    }
                }
            }
        }
        if(maxdiff > 0)
        {
            std::cout << "Max diff: " << maxdiff << std::endl;
            std::cout << "gpu[" << mn << ", " << mc << ", " << md << ", " << mh << ", " << mw
                      << "]: " << gpuout(mn, mc, md, mh, mw) << " :: ";
            std::cout << "cpu[" << mn << ", " << mc << ", " << md << ", " << mh << ", " << mw
                      << "]: " << cpuout(mn, mc, md, mh, mw) << std::endl;
        }
#else
#if(MIO_BN_SP_TEST_DEBUG == 1)
        std::cout << "Running back propagation spatial with S set." << std::endl;
#endif
        verify(verify_backward_3d_bn_spatial_use_saved<T, PREC_TYPE>{
            input, dy_input, scale, savedMean, savedInvVar});
#endif
    }
};

int main(int argc, const char* argv[])
{
#if(MIO_BN_TIME_EVERYTHING == 1)
    auto t_start = std::chrono::high_resolution_clock::now();
#endif
    test_drive<batch_norm_3d_spatial_driver>(argc, argv);

#if(MIO_BN_TIME_EVERYTHING == 1)
    auto t_end = std::chrono::high_resolution_clock::now();

    std::cout << "Wall clock: full SPATIAL test pass time: "
              << std::chrono::duration<double>(t_end - t_start).count() << " seconds." << std::endl;
#endif
    return 0;
}
