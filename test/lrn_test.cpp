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
#include "driver.hpp"
#include "test.hpp"
#include "verify.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/lrn.hpp>
#include <random>
#include <algorithm>
#include <iterator>
#include <limits>
#include <iostream>

template <class T>
struct verify_lrn_foward
{
    miopen::LRNDescriptor lrn;
    tensor<T> input;

    tensor<T> cpu()
    {
        auto output = input;
        int n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(input.desc.GetLengths());

        auto alpha  = lrn.GetAlpha();
        auto beta   = lrn.GetBeta();
        auto K      = lrn.GetK();
        auto lrn_n  = lrn.GetN();
        auto radius = (lrn.GetN() - 1) / 2;
        auto mode   = lrn.GetMode();

        if(mode == miopenLRNCrossChannel)
        {
            auto alphaoverarea = alpha / lrn_n;

            par_ford(n_batch, height, width)([&](int b, int h, int w) {
                double scale = 0;
                ford(channels)([&](int c) {
                    auto start = (c - radius) < 0 ? 0 : (c - radius);
                    auto end   = (c + radius) > channels ? channels : (c + radius);

                    for(auto k = start; k < end; k++)
                    {
                        scale += std::pow(input(b, k, h, w), 2);
                    }

                    scale *= alphaoverarea;
                    scale += K;
                    scale = std::pow(scale, -beta);

                    output(b, c, h, w) = input(b, c, h, w) * scale;
                });
            });
        }
        else
        {

            par_ford(n_batch, channels)([&](int b, int c) {
                double scale = 0;
                ford(height, width)([&](int h, int w) {
                    auto left          = (w - radius) < 0 ? 0 : (w - radius);
                    auto right         = (w + radius) > width ? width : (w + radius);
                    auto top           = (h - radius) < 0 ? 0 : (h - radius);
                    auto bottom        = (h + radius) > height ? height : (h + radius);
                    auto alphaoverarea = alpha / ((right - left) * (bottom - top));

                    for(auto i = left; i < right; i++)
                    {
                        for(auto j = top; j < bottom; j++)
                        {
                            scale += std::pow(input(b, c, h, w), 2);
                        }
                    }
                    scale *= alphaoverarea;
                    scale += K;
                    scale = std::pow(scale, -beta);
                    output(b, c, h, w) = input(b, c, h, w) * scale;
                });
            });
        }

        return output;
    }

    tensor<T> gpu()
    {
        auto&& handle = get_handle();
        auto out      = input;
        auto in_dev   = handle.Write(input.data);
        auto out_dev  = handle.Write(out.data);
        auto alpha    = lrn.GetAlpha();
        auto beta     = lrn.GetBeta();
        auto bDoBwd   = false;

        lrn.Forward(handle,
                    &alpha,
                    input.desc,
                    in_dev.get(),
                    &beta,
                    out.desc,
                    out_dev.get(),
                    bDoBwd,
                    nullptr);

        out.data = handle.Read<T>(out_dev, out.data.size());
        return out;
    }

    void fail(int)
    {
        std::cout << "verify_lrn_foward" << std::endl;
        std::cout << "Input Tensor"
                  << " " << input.desc.ToString() << std::endl;
    }
};

template <class T>
struct verify_lrn_bwd
{

    miopen::LRNDescriptor lrn;
    tensor<T> inputY;
    tensor<T> inputDY;
    tensor<T> inputX; // scale
    tensor<T> inputDX;

    tensor<T> cpu()
    {

        auto outputDX = inputDY;
        auto scale    = inputY;

        std::fill(scale.begin(), scale.end(), 0);

        int n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(inputY.desc.GetLengths());

        auto alpha  = lrn.GetAlpha();
        auto beta   = lrn.GetBeta();
        auto lrn_n  = lrn.GetN();
        auto K      = lrn.GetK();
        auto mode   = lrn.GetMode();
        auto radius = (lrn_n - 1) / 2;

        if(mode == miopenLRNWithinChannel)
        {
            for(auto b = 0; b < n_batch; b++)
            {
                for(auto c = 0; c < channels; c++)
                {
                    for(auto h = 0; h < height; h++)
                    {
                        for(auto w = 0; w < width; w++)
                        {
                            double ydy = 0;

                            // change to zero padding
                            auto left        = (w - radius) < 0 ? 0 : (w - radius);
                            auto right       = (left + lrn_n) > width ? width : (left + lrn_n);
                            auto top         = (h - radius) < 0 ? 0 : (h - radius);
                            auto bottom      = (top + lrn_n) > height ? height : (top + lrn_n);
                            auto adjust_area = (right - left) * (bottom - top);
                            auto cache_ratio_value = 2 * alpha * beta / adjust_area;
                            auto alpha_over_area   = alpha / adjust_area;

                            for(auto i = left; i < right; i++)
                            {
                                for(auto j = top; j < bottom; j++)
                                {
                                    scale(b, c, j, i) += inputY(b, c, j, i) * inputY(b, c, j, i);
                                    scale(b, c, j, i) += K;
                                    scale(b, c, j, i) *= alpha_over_area;
                                }
                            }

                            for(auto i = left; i < right; i++)
                            {
                                for(auto j = top; j < bottom; j++)
                                {
                                    ydy += inputY(b, c, j, i) * inputDY(b, c, j, i) /
                                           scale(b, c, j, i);
                                }
                            }

                            outputDX(b, c, h, w) =
                                pow(scale(b, c, h, w), -beta) * inputDY(b, c, h, w) -
                                cache_ratio_value * inputX(b, c, h, w) * ydy;
                        }
                    }
                }
            }
        }
        else
        {

            auto cache_ratio_value = 2 * alpha * beta / lrn_n;

            for(auto b = 0; b < n_batch; b++)
            {
                for(auto h = 0; h < height; h++)
                {
                    for(auto w = 0; w < width; w++)
                    {
                        double scale_x = 0;
                        double ydy     = 0;

                        for(auto c = 0; c < channels; c++)
                        {
                            auto intensityX = inputX(b, c, h, w);
                            ydy += inputY(b, c, h, w) * inputDY(b, c, h, w) / scale_x;
                        }

                        for(auto c = 0; c < channels; c++)
                            outputDX(b, c, h, w) = pow(scale_x, -beta) * inputDY(b, c, h, w) -
                                                   cache_ratio_value * inputX(b, c, h, w) * ydy;
                    }
                }
            }
        }

        return outputDX;
    }

    tensor<T> gpu()
    {
        auto&& handle      = get_handle();
        auto dinput        = inputY;
        auto in_dev        = handle.Write(inputY.data);
        auto dout_dev      = handle.Write(inputDY.data);
        auto out_dev       = handle.Write(inputX.data);
        auto din_dev       = handle.Create<T>(dinput.data.size());
        auto workspace_dev = handle.Create<T>(inputY.data.size());

        auto alpha = lrn.GetAlpha(), beta = lrn.GetBeta();
        lrn.Backward(handle,
                     &alpha,
                     // y
                     inputY.desc,
                     in_dev.get(),
                     // dy
                     inputDY.desc,
                     dout_dev.get(),
                     // x
                     inputX.desc,
                     out_dev.get(),
                     &beta,
                     // dx
                     dinput.desc,
                     din_dev.get(),

                     workspace_dev.get());

        dinput.data = handle.Read<T>(din_dev, dinput.data.size());
        return dinput;
    }

    void fail(int)
    {
        std::cout << "verify_lrn_bwd" << std::endl;
        std::cout << "Input Tensor Y"
                  << " " << inputY.desc.ToString() << std::endl;
        std::cout << "Input Tensor DY"
                  << " " << inputDY.desc.ToString() << std::endl;
        std::cout << "Input Tensor X"
                  << " " << inputX.desc.ToString() << std::endl;
    }
};

template <class T>
struct lrn_driver : test_driver
{
    tensor<T> input;

    unsigned int n = 0;
    T alpha        = 0;
    T beta         = 0;
    T k            = 0;
    std::string mode;

    std::unordered_map<std::string, miopenLRNMode_t> mode_lookup = {
        {"WITHIN_CHANNEL", miopenLRNWithinChannel}, {"ACROSS_CHANNEL", miopenLRNCrossChannel}};

    lrn_driver()
    {
        add(input, "input", get_input_tensor());
        add(n, "N", generate_data({1, 3, 5}));
        add(alpha, "alpha", generate_data({1.0}));
        add(beta, "beta", generate_data({0}));
        add(k, "K", generate_data({1}));
        add(mode, "mode", generate_data({"Within_Channel", "Across_Channel"}));
    }

    ~lrn_driver() {}

    void run()
    {
        miopen::LRNDescriptor lrn{mode_lookup.at(miopen::ToUpper(mode)), n, {alpha, beta, k}};

        auto fwd_output = verify(verify_lrn_foward<T>{lrn, input});

        std::cout << "fwd_output = " << fwd_output.first.desc.ToString() << std::endl;

        auto dout = fwd_output.first;
        dout.generate([&](int b, int c, int h, int w) {
            T x      = fwd_output.first(b, c, h, w);
            double y = (877 * b + 547 * c + 701 * h + 1049 * w + static_cast<int>(769 * x)) % 2503;
            return ((x * y) / 1301.0);
        });

        // auto bwd_output = verify(verify_lrn_bwd<T>{lrn, input, dout, fwd_output.first});
    };
};

int main(int argc, const char* argv[]) { test_drive<lrn_driver<double>>(argc, argv); };
