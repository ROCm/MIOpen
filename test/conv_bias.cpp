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
#include <miopen/stringutils.hpp>
#include <utility>

#include "network_data.hpp"
#include "driver.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include "cpu_bias.hpp"

struct scalar_gen_random_integer
{
    unsigned long min_val = 1;
    unsigned long max_val = 16;

    double operator()() const
    {
        return static_cast<double>(min_val + std::rand() % (max_val - min_val + 1));
    }
};

template <class T>
struct verify_backwards_bias
{
    tensor<T> output;
    tensor<T> bias;

    tensor<T> cpu() const
    {
        auto rbias = bias;
        cpu_bias_backward_data(output, rbias);
        return rbias;
    }

    tensor<T> gpu() const
    {
        auto&& handle = get_handle();
        auto rbias    = bias;

        auto out_dev  = handle.Write(output.data);
        auto bias_dev = handle.Write(rbias.data);

        float alpha = 1, beta = 0;
        ConvolutionBackwardBias(
            handle, &alpha, output.desc, out_dev.get(), &beta, rbias.desc, bias_dev.get());

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

template <class T, std::size_t ConvDim>
struct conv_bias_driver : test_driver
{
    tensor<T> output;

    void run()
    {
        std::vector<std::size_t> bias_lens(2 + ConvDim, 1);
        bias_lens[1] = output.desc.GetLengths()[1];

        tensor<T> bias(bias_lens);

        verify(verify_backwards_bias<T>{output, bias});
    }
};

template <class T>
struct conv2d_bias_driver : public conv_bias_driver<T, 2>
{
    std::string conv_dim_type;

    conv2d_bias_driver()
    {
        this->add(conv_dim_type, "conv_dim_type", this->generate_data({"conv2d"}));

        auto gen_value = [](auto... is) {
            return scalar_gen_random_integer{1, miopen_type<T>{} == miopenHalf ? 5 : 17}() *
                   tensor_elem_gen_checkboard_sign{}(is...);
        };

        this->add(this->output, "output", this->get_tensor(get_inputs, gen_value));
    }
};

template <class T>
struct conv3d_bias_driver : public conv_bias_driver<T, 3>
{
    std::string conv_dim_type;

    conv3d_bias_driver()
    {
        this->add(conv_dim_type, "conv_dim_type", this->generate_data({"conv3d"}));

        auto gen_value = [](auto... is) {
            return scalar_gen_random_integer{1, miopen_type<T>{} == miopenHalf ? 5 : 17}() *
                   tensor_elem_gen_checkboard_sign{}(is...);
        };

        this->add(this->output, "output", this->get_tensor(get_3d_conv_input_shapes, gen_value));
    }
};

int main(int argc, const char* argv[])
{
    std::vector<std::string> as(argv + 1, argv + argc);

    bool do_conv2d = std::any_of(as.begin(), as.end(), [](auto&& arg) { return arg == "conv2d"; });
    bool do_conv3d = std::any_of(as.begin(), as.end(), [](auto&& arg) { return arg == "conv3d"; });
    bool do_all    = std::any_of(as.begin(), as.end(), [](auto&& arg) { return arg == "--all"; });

    if(do_conv2d and !do_conv3d)
    {
        test_drive<conv2d_bias_driver>(argc, argv);
    }
    else if(!do_conv2d and do_conv3d)
    {
        test_drive<conv3d_bias_driver>(argc, argv);
    }
    else if((do_conv2d and do_conv3d) or do_all)
    {
        test_drive<conv2d_bias_driver>(argc, argv);
        test_drive<conv3d_bias_driver>(argc, argv);
    }
    else
    {
        test_drive<conv2d_bias_driver>(argc, argv);
    }
}
