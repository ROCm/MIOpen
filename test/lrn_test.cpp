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

    verify_lrn_foward(const miopen::LRNDescriptor& plrnDesc, const tensor<T>& pinput)
    {
        lrn   = plrnDesc;
        input = pinput;
    }

    tensor<T> cpu() const
    {
        auto output = tensor<T>{input.desc.GetLengths()};
        int n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(input.desc.GetLengths());

        auto alpha       = lrn.GetAlpha();
        auto beta        = lrn.GetBeta();
        auto K           = lrn.GetK();
        auto lrn_n       = lrn.GetN();
        int radius_lower = static_cast<int>((lrn_n - 1) / 2);
        int radius_upper = static_cast<int>(lrn_n / 2);
        auto mode        = lrn.GetMode();

        if(mode == miopenLRNCrossChannel)
        {
            auto alphaoverarea = alpha / lrn_n;
            par_ford(n_batch, channels, height, width)([&](int b, int c, int h, int w) {
                int start = c < radius_lower ? 0 : (c - radius_lower);
                int end   = (c + radius_upper + 1) > channels ? channels : (c + radius_upper + 1);

                double scale = 0;
                for(int k = start; k < end; k++)
                {
                    scale += std::pow(input(b, k, h, w), 2);
                }

                scale *= alphaoverarea;
                scale += K;
                scale = std::pow(scale, -beta);

                output(b, c, h, w) = static_cast<T>(scale * input(b, c, h, w));
            });
        }
        else
        {
            double alphaoverarea = radius_upper == 0 ? 1 : alpha / (lrn_n * lrn_n);
            par_ford(n_batch, channels, height, width)([&](int b, int c, int h, int w) {
                double scale = 0;
                int left     = (w - radius_lower) < 0 ? 0 : (w - radius_lower);
                int right    = (w + radius_upper + 1) > width ? width : (w + radius_upper + 1);
                int top      = (h - radius_lower) < 0 ? 0 : (h - radius_lower);
                int bottom   = (h + radius_upper + 1) > height ? height : (h + radius_upper + 1);

                for(int i = left; i < right; i++)
                {
                    for(int j = top; j < bottom; j++)
                    {
                        scale += std::pow(input(b, c, j, i), 2);
                    }
                }
                scale *= alphaoverarea;
                scale += K;
                scale              = std::pow(scale, -beta);
                output(b, c, h, w) = static_cast<T>(scale * input(b, c, h, w));
            });
        }

        return output;
    }

    tensor<T> gpu() const
    {
        auto&& handle = get_handle();
        auto out      = tensor<T>{input.desc.GetLengths()};
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

    void fail(int) const
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
    tensor<T> scale;

    verify_lrn_bwd(const miopen::LRNDescriptor& plrn,
                   const tensor<T>& pout,
                   const tensor<T>& pdout,
                   const tensor<T>& pin,
                   const tensor<T>& pscale)
    {
        lrn     = plrn;
        inputY  = pout;
        inputDY = pdout;
        inputX  = pin;
        scale   = pscale;
    }

    tensor<T> cpu() const
    {
        auto routputDX = tensor<T>{inputX.desc.GetLengths()};
        int n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(inputY.desc.GetLengths());

        auto alpha       = lrn.GetAlpha();
        auto beta        = lrn.GetBeta();
        auto lrn_n       = lrn.GetN();
        auto mode        = lrn.GetMode();
        int radius_lower = static_cast<int>((lrn_n - 1) / 2);
        int radius_upper = static_cast<int>(lrn_n / 2);

        if(mode == miopenLRNWithinChannel)
        {
            auto adjust_area       = lrn_n * lrn_n;
            auto cache_ratio_value = 2 * alpha * beta / adjust_area;

            par_ford(n_batch, channels, height, width)([&](int b, int c, int h, int w) {
                int left   = w < radius_upper ? 0 : (w - radius_upper);
                int right  = (w + radius_lower + 1) > width ? width : (w + radius_lower + 1);
                int top    = h < radius_upper ? 0 : (h - radius_upper);
                int bottom = (h + radius_lower + 1) > height ? height : (h + radius_lower + 1);

                double ydy = 0;
                for(int i = left; i < right; i++)
                {
                    for(int j = top; j < bottom; j++)
                    {
                        ydy += (double(inputY(b, c, j, i) * inputDY(b, c, j, i)) /
                                double(scale(b, c, j, i)));
                    }
                }

                routputDX(b, c, h, w) = static_cast<T>(
                    std::pow(static_cast<double>(scale(b, c, h, w)), -beta) * inputDY(b, c, h, w) -
                    cache_ratio_value * inputX(b, c, h, w) * ydy);
            });
        }
        else
        {
            auto cache_ratio_value = 2 * alpha * beta / lrn_n;

            par_ford(n_batch, channels, height, width)([&](int b, int c, int h, int w) {
                int start = c < radius_upper ? 0 : (c - radius_upper);
                int end   = (c + radius_lower + 1) > channels ? channels : (c + radius_lower + 1);

                double ydy = 0;
                for(auto k = start; k < end; k++)
                {
                    ydy += (double(inputY(b, k, h, w) * inputDY(b, k, h, w)) /
                            double(scale(b, k, h, w)));
                }

                routputDX(b, c, h, w) = static_cast<T>(
                    std::pow(static_cast<double>(scale(b, c, h, w)), -beta) * inputDY(b, c, h, w) -
                    cache_ratio_value * inputX(b, c, h, w) * ydy);
            });
        }

        return routputDX;
    }

    tensor<T> gpu() const
    {
        auto&& handle     = get_handle();
        auto routputDX    = tensor<T>{inputX.desc.GetLengths()};
        auto inputY_dev   = handle.Write(inputY.data);
        auto inputDY_dev  = handle.Write(inputDY.data);
        auto inputX_dev   = handle.Write(inputX.data);
        auto outputDX_dev = handle.Create<T>(routputDX.data.size());
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
                     routputDX.desc, // DX
                     outputDX_dev.get(),
                     scale_dev.get());

        routputDX.data = handle.Read<T>(outputDX_dev, routputDX.data.size());
        return routputDX;
    }

    void fail(int) const
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

    unsigned int n = 1;
    double alpha   = 1;
    double beta    = 1;
    double k       = 1;
    std::string mode;

    std::unordered_map<std::string, miopenLRNMode_t> mode_lookup = {
        {"WITHIN_CHANNEL", miopenLRNWithinChannel}, {"ACROSS_CHANNEL", miopenLRNCrossChannel}};

    lrn_driver()
    {
        add(input,
            "input",
            get_input_tensor(tensor_elem_gen_integer{miopen_type<T>{} == miopenHalf ? 5 : 17}));
        add(n, "N", generate_data({1, 4, 5}));
        add(alpha, "alpha", generate_data({double(1)}));
        add(beta, "beta", generate_data({double(1)}));
        add(k, "K", generate_data({double(1)}));
        add(mode, "mode", generate_data({"Within_Channel", "Across_Channel"}));
    }

    void run()
    {
        std::size_t n_batch, channels, height, width;
        std::tie(n_batch, channels, height, width) = miopen::tien<4>(input.desc.GetLengths());
        size_t total_mem  = 5 * input.desc.GetNumBytes(); // estimate based on backward pass
        size_t device_mem = get_handle().GetGlobalMemorySize();
        if(total_mem >= device_mem)
        {
            show_command();
            std::cout << "Config requires " << total_mem
                      << " Bytes to write all necessary tensors to GPU. GPU has " << device_mem
                      << " Bytes of memory." << std::endl;
            return;
        }

        miopen::LRNDescriptor lrn{mode_lookup.at(miopen::ToUpper(mode)), n, {alpha, beta, k}};

        auto out                = verify(verify_lrn_foward<T>{lrn, input});
        unsigned long max_value = miopen_type<T>{} == miopenHalf ? 5 : 17;

        auto scale = tensor<T>{n_batch, channels, height, width}.generate(
            tensor_elem_gen_integer{max_value});
        auto dout = tensor<T>{n_batch, channels, height, width}.generate(
            tensor_elem_gen_integer{max_value});
        par_ford(n_batch, channels, height, width)(
            [&](int b, int c, int h, int w) { scale(b, c, h, w) += 1; });

        verify(verify_lrn_bwd<T>{lrn, out.first, dout, input, scale});
    };
};

// To address compiler issue for bfloat1 type in SWDEV-202752
// creating explicit instance of lrn_driver with bfloat16 with noop
template <>
struct lrn_driver<bfloat16> : test_driver
{
    lrn_driver() {}
    void run() { std::cout << "bfloat16 is not supported in lrn" << std::endl; };
};

int main(int argc, const char* argv[]) { test_drive<lrn_driver>(argc, argv); };
