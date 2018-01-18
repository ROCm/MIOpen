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
#include "test.hpp"
#include <array>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <miopen/convolution.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <utility>

// #include "network_data.hpp"
#include "driver.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <miopen/stringutils.hpp>

template <class T>
struct verify_backwards_bias
{
    tensor<T> output;
    tensor<T> bias;
    miopenConvolutionMode_t mode;

    tensor<T> cpu() const
    {
        auto rbias = bias;
        int out_n, out_c, out_h, out_w;
        std::tie(out_n, out_c, out_h, out_w) = miopen::tien<4>(output.desc.GetLengths());

        par_ford(out_c)([&](int c)
        {
            double acc = 0;
            ford(out_n, out_h, out_w)([&](int n, int h, int w)
            {
                acc += output(n, c, h, w);
            });
            rbias[c] = acc;
        });
        return rbias;
    }

    tensor<T> gpu() const
    {
        auto&& handle = get_handle();
        auto rbias = bias;

        auto out_dev = handle.Write(output.data);
        auto bias_dev = handle.Write(rbias.data);

        float alpha = 1, beta = 0;
        ConvolutionBackwardBias(handle,
                                         &alpha,
                                         rbias.desc,
                                         bias_dev.get(),
                                         &beta,
                                         output.desc,
                                         out_dev.get());


        rbias.data = handle.Read<T>(bias_dev, rbias.data.size());
        return rbias;
    }

    void fail(int = 0) const
    {
        std::cout << "Backwards bias: " << std::endl;
        std::cout << "Output tensor: " << output.desc.ToString() << std::endl;
        std::cout << "Bias tensor: " << bias.desc.ToString() << std::endl;
    }
};

template <class T>
struct conv_bias_driver : test_driver
{
    tensor<T> output;
    std::string conv_mode;

    std::unordered_map<std::string, miopenConvolutionMode_t> cmode_lookup = {
        {"CONV", miopenConvolution}, {"TRANS", miopenTranspose}};

    conv_bias_driver()
    {
        add(output, "output", get_input_tensor());
    }

    void run()
    {
        auto mode        = cmode_lookup[miopen::ToUpper(conv_mode)];
        tensor<T> bias = {1, output.desc.GetLengths()[1], 1, 1};
        verify(verify_backwards_bias<T>{output, bias, mode});
    }
};

int main(int argc, const char* argv[]) { test_drive<conv_bias_driver>(argc, argv); }
