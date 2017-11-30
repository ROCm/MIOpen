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

#include "test.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <miopen/convolution.hpp>
#include <miopen/miopen.h>
#include <miopen/softmax.hpp>
#include <miopen/tensor.hpp>
#include <utility>

#include "driver.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

struct verify_forward_sofmax
{
    template <class T>
    tensor<T> cpu(const tensor<T>& input)
    {
        auto out = input;
        std::fill(out.begin(), out.end(), 0);

        int in_n, in_c, in_h, in_w;
        std::tie(in_n, in_c, in_h, in_w) = miopen::tien<4>(input.desc.GetLengths());

        par_ford(in_n, in_h, in_w)([&](int o, int i, int j) {
            T max_c = std::numeric_limits<T>::lowest();
            ford(in_c)([&](int w) { max_c = std::max(max_c, input(o, w, i, j)); });

            T sum = 0;
            ford(in_c)([&](int w) { sum += std::exp(input(o, w, i, j) - max_c); });

            ford(in_c)([&](int w) { out(o, w, i, j) = std::exp(input(o, w, i, j) - max_c) / sum; });

        });
        return out;
    }

    template <class T>
    tensor<T> gpu(const tensor<T>& input)
    {
        auto&& handle = get_handle();
        auto out      = input;

        auto out_dev = handle.Write(out.data);

        float alpha = 1, beta = 0;

        miopen::SoftmaxForward(handle, &alpha, &beta, input.desc, out_dev.get());

        out.data = handle.Read<T>(out_dev, out.data.size());
        return out;
    }

    template <class T>
    void fail(float, const tensor<T>& input)
    {
        std::cout << "Forward Sofmax: " << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
    }
};

struct verify_backward_sofmax
{
    template <class T>
    tensor<T> cpu(const tensor<T>& out, const tensor<T>& dout)
    {
        auto input = dout;

        int in_n, in_c, in_h, in_w;
        std::tie(in_n, in_c, in_h, in_w) = miopen::tien<4>(input.desc.GetLengths());

        par_ford(in_n, in_h, in_w)([&](int o, int i, int j) {
            T sum = 0;
            ford(in_c)([&](int c) { sum += out(o, c, i, j) * dout(o, c, i, j); });

            ford(in_c)(
                [&](int c) { input(o, c, i, j) = out(o, c, i, j) * (input(o, c, i, j) - sum); });
        });
        return input;
    }

    template <class T>
    tensor<T> gpu(const tensor<T>& out, const tensor<T>& dout)
    {
        auto&& handle = get_handle();
        auto input    = dout;

        auto in_dev  = handle.Write(input.data);
        auto out_dev = handle.Write(out.data);

        float alpha = 1, beta = 0;

        miopen::SoftmaxBackward(
            handle, &alpha, out.desc, out_dev.get(), &beta, input.desc, in_dev.get());

        input.data = handle.Read<T>(in_dev, input.data.size());
        return input;
    }

    template <class T>
    void fail(float, const tensor<T>& output, const tensor<T>&)
    {
        std::cout << "Backward Sofmax: " << std::endl;
        std::cout << "Output tensor: " << output.desc.ToString() << std::endl;
    }
};

template <class T>
struct softmax_driver : test_driver
{
    tensor<T> input;

    softmax_driver() { add(input, "input", get_input_tensor()); }

    void run()
    {
        auto out  = verify(verify_forward_sofmax{}, input);
        auto dout = input;
        dout.generate([&](int n, int c, int h, int w) {
            T x      = input(n, c, h, w);
            double y = (877 * n + 547 * c + 701 * h + 1049 * w + static_cast<int>(769 * x)) % 2503;
            return ((x * y) / 1301.0);
        });
        verify(verify_backward_sofmax{}, out.first, dout);
    }
};

int main(int argc, const char* argv[]) { test_drive<softmax_driver<float>>(argc, argv); }
