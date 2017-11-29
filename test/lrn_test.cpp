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

        CHECK((lrn_n & 1) == 1);
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
                    auto left   = (w - radius) < 0 ? 0 : (w - radius);
                    auto right  = (w + radius) > width ? width : (w + radius);
                    auto top    = (h - radius) < 0 ? 0 : (h - radius);
                    auto bottom = (h + radius) > height ? height : (h + radius);
                    auto alphaoverarea =
                        radius == 0 ? 0 : alpha / ((right - left) * (bottom - top));

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
    tensor<T> inputX;
    tensor<T> outputDX;
    tensor<T> scale;

    tensor<T> cpu()
    {
        int n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(inputY.desc.GetLengths());

        auto alpha  = lrn.GetAlpha();
        auto beta   = lrn.GetBeta();
        auto lrn_n  = lrn.GetN();
        auto mode   = lrn.GetMode();
        auto radius = (lrn_n - 1) / 2;

        if(mode == miopenLRNWithinChannel)
        {
            par_ford(n_batch, channels)([&](int b, int c) {
                ford(height, width)([&](int h, int w) {
                    double ydy             = 0;
                    auto left              = (w - radius) < 0 ? 0 : (w - radius);
                    auto right             = (left + lrn_n) > width ? width : (left + lrn_n);
                    auto top               = (h - radius) < 0 ? 0 : (h - radius);
                    auto bottom            = (top + lrn_n) > height ? height : (top + lrn_n);
                    auto adjust_area       = (right - left) * (bottom - top);
                    auto cache_ratio_value = 2 * alpha * beta / adjust_area;

                    for(auto i = left; i < right; i++)
                    {
                        for(auto j = top; j < bottom; j++)
                        {
                            ydy += (inputY(b, c, j, i) * inputDY(b, c, j, i) / scale(b, c, j, i));
                        }
                    }

                    outputDX(b, c, h, w) = pow(scale(b, c, h, w), -beta) * inputDY(b, c, h, w) -
                                           cache_ratio_value * inputX(b, c, h, w) * ydy;
                });
            });
        }
        else
        {
            auto cache_ratio_value = 2 * alpha * beta / lrn_n;

            par_ford(n_batch, height, width)([&](int b, int h, int w) {
                ford(channels)([&](int c) {
                    double ydy = 0;
                    auto start = (c - radius) < 0 ? 0 : (c - radius);
                    auto end   = (c + radius) > channels ? channels : (c + radius);

                    for(auto k = start; k < end; k++)
                    {
                        ydy += (inputY(b, k, h, w) * inputDY(b, k, h, w) / scale(b, k, h, w));
                    }

                    outputDX(b, c, h, w) = pow(scale(b, c, h, w), -beta) * inputDY(b, c, h, w) -
                                           cache_ratio_value * inputX(b, c, h, w) * ydy;
                });
            });
        }

        return outputDX;
    }

    tensor<T> gpu()
    {
        auto&& handle     = get_handle();
        auto inputY_dev   = handle.Write(inputY.data);
        auto inputDY_dev  = handle.Write(inputDY.data);
        auto inputX_dev   = handle.Write(inputX.data);
        auto outputDX_dev = handle.Create<T>(outputDX.data.size());
        auto scale_dev    = handle.Write(scale.data);

        auto alpha = lrn.GetAlpha(), beta = lrn.GetBeta();
        lrn.Backward(handle,
                     &alpha,
                     inputY.desc, // Y
                     inputY_dev.get(),
                     inputDY.desc, // DY
                     inputDY_dev.get(),
                     inputX.desc, // X
                     inputX_dev.get(),
                     &beta,
                     outputDX.desc, // DX
                     outputDX_dev.get(),
                     scale_dev.get());

        outputDX.data = handle.Read<T>(outputDX_dev, outputDX.data.size());
        return outputDX;
    }

    void fail(int)
    {
        std::cout << "verify_lrn_bwd" << std::endl;
        std::cout << "Input Tensor Y"
                  << " " << inputY.desc.ToString() << std::endl;
        std::cout << "Input Tensor DY"
                  << " " << inputDY.desc.ToString() << std::endl;
        std::cout << "Input Tensor X"
                  << " " << scale.desc.ToString() << std::endl;
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

    void run()
    {
        miopen::LRNDescriptor lrn{mode_lookup.at(miopen::ToUpper(mode)), n, {alpha, beta, k}};

        auto OutputDX   = input;
        auto fwd_output = verify(verify_lrn_foward<T>{lrn, input});
        auto out        = fwd_output.first;

        std::size_t n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(input.desc.GetLengths());
        auto scale  = tensor<T>{n_batch, channels, height, width}.generate(rand_gen{});
        auto inputX = tensor<T>{n_batch, channels, height, width}.generate(rand_gen{});
        par_ford(n_batch, channels, height, width)(
            [&](int b, int c, int h, int w) { scale(b, c, h, w) += 1; });

        auto bwd_output = verify(verify_lrn_bwd<T>{lrn, input, out, inputX, OutputDX, scale});
    };
};

int main(int argc, const char* argv[]) { test_drive<lrn_driver<float>>(argc, argv); };
