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
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <miopen/activ.hpp>
#include <miopen/miopen.h>
#include <miopen/stringutils.hpp>
#include <miopen/tensor.hpp>
#include <utility>

#include "driver.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

std::string to_name(miopenActivationMode_t m)
{
#define STRING_CASE(x) \
    case x: return #x; break;
    switch(m)
    {
        STRING_CASE(miopenActivationPASTHRU)
        STRING_CASE(miopenActivationLOGISTIC)
        STRING_CASE(miopenActivationTANH)
        STRING_CASE(miopenActivationRELU)
        STRING_CASE(miopenActivationSOFTRELU)
        STRING_CASE(miopenActivationABS)
        STRING_CASE(miopenActivationPOWER)
        STRING_CASE(miopenActivationCLIPPEDRELU)
        STRING_CASE(miopenActivationLEAKYRELU)
        STRING_CASE(miopenActivationELU)
    }
    return "";
}

template <class T>
struct verify_forward_activation
{
    tensor<T> input;
    miopen::ActivationDescriptor desc;

    template <class A>
    tensor<T> cpu(A a)
    {
        auto out = input;

        input.par_for_each(
            [&](int o, int w, int i, int j) { out(o, w, i, j) = a(input(o, w, i, j)); });

        return out;
    }

    template <class A>
    tensor<T> gpu(A)
    {
        auto&& handle = get_handle();
        auto out      = input;
        auto in_dev   = handle.Write(input.data);
        auto out_dev  = handle.Write(out.data);

        float alpha = 1, beta = 0;

        desc.Forward(handle, &alpha, input.desc, in_dev.get(), &beta, out.desc, out_dev.get());

        out.data = handle.Read<T>(out_dev, out.data.size());
        return out;
    }

    template <class A>
    void fail(float, A)
    {
        std::cout << "Forward Activation: " << to_name(desc.GetMode()) << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
    }
};

template <class T>
struct verify_backwards_activation
{
    tensor<T> input;
    tensor<T> dout;
    tensor<T> out;
    miopen::ActivationDescriptor desc;

    template <class A>
    tensor<T> cpu(A a)
    {
        auto dinput = input;

        input.par_for_each([&](int o, int w, int i, int j) {
            dinput(o, w, i, j) = a(dout(o, w, i, j), input(o, w, i, j), out(o, w, i, j));
        });

        return dinput;
    }

    template <class A>
    tensor<T> gpu(A)
    {
        auto&& handle = get_handle();
        auto dinput   = input;

        auto in_dev   = handle.Write(input.data);
        auto dout_dev = handle.Write(dout.data);
        auto out_dev  = handle.Write(out.data);
        auto din_dev  = handle.Write(dinput.data);

        float alpha = 1, beta = 0;

        desc.Forward(handle, &alpha, input.desc, in_dev.get(), &beta, out.desc, out_dev.get());
        desc.Backward(handle,
                      &alpha,
                      // y
                      out.desc,
                      out_dev.get(),
                      // dy
                      dout.desc,
                      dout_dev.get(),
                      // x
                      input.desc,
                      in_dev.get(),
                      &beta,
                      // dx
                      dinput.desc,
                      din_dev.get());

        dinput.data = handle.Read<T>(din_dev, dinput.data.size());
        return dinput;
    }

    template <class A>
    void fail(float, A)
    {
        std::cout << "Backwards Activation: " << to_name(desc.GetMode()) << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
    }
};

struct select_first
{
    template <class T>
    auto operator()(const T& x) MIOPEN_RETURNS(x.first); // NOLINT (readability-const-return-type)
};

template <class T>
struct activation_driver : test_driver
{
    tensor<T> input;
    double alpha     = 0.95;
    double beta      = 2.3;
    double gamma     = 3.4;
    std::string mode = "PASTHRU";
    std::unordered_map<std::string, std::function<void()>> lookup;
    bool packed = true;

    template <class A>
    struct callback
    {
        void operator()(activation_driver* self) const { self->template run<A>(); }
    };

    template <class Forward, class Backward>
    void add_mode(miopenActivationMode_t m, Forward f, Backward b)
    {
        lookup.emplace(transform_mode(to_name(m)), [=] { this->run(m, f, b); });
    }

    activation_driver()
    {
        disabled_cache = true;
        add_mode(
            miopenActivationPASTHRU,
            [=](double x) { return x; },
            [=](double dy, double, double) { return dy; });
        add_mode(
            miopenActivationLOGISTIC,
            [=](double x) { return 1 / (1 + std::exp(-x)); },
            [=](double dy, double, double y) { return dy * y * (1 - y); });
        add_mode(
            miopenActivationTANH,
            // y = beta * tanh(alpha * x)
            [=](double x) { return beta * std::tanh(alpha * x); },
            [=](double dy, double, double y) { return dy * alpha * (beta - y * y / beta); });
        add_mode(
            miopenActivationRELU,
            [=](double x) { return (x > 0) ? x : 0; },
            [=](double dy, double x, double) { return (x > 0) ? dy : 0; });
        add_mode(
            miopenActivationSOFTRELU,
            [=](double x) { return std::log1p(std::exp(x)); },
            [=](double dy, double x, double) {
                static const double threshold = 50.;
                double expval                 = std::exp(std::min(x, threshold));
                return dy * expval / (expval + 1.0);
            });
        add_mode(
            miopenActivationABS,
            [=](double x) { return std::abs(x); },
            [=](double dy, double x, double) { return dy * ((x > 0) ? 1 : -1); });
        add_mode(
            miopenActivationPOWER,
            [=](double x) {
                double v = alpha + beta * x;
                return v <= std::numeric_limits<double>::epsilon() ? 0 : pow(v, gamma);
            },
            [=](double, double x, double y) {
                auto v = alpha + beta * x;
                return v <= std::numeric_limits<double>::epsilon() ? 0 : gamma * beta * y / v;
            });
        add_mode(
            miopenActivationCLIPPEDRELU,
            [=](double x) { return std::min(alpha, std::max(double(0), x)); },
            [=](double dy, double x, double) { return (x > 0 && x <= alpha) ? dy : 0; });
        add_mode(
            miopenActivationLEAKYRELU,
            [=](double x) { return (x > 0) ? x : x * alpha; },
            [=](double dy, double x, double) { return dy * ((x > 0) ? 1 : alpha); });
        add_mode(
            miopenActivationELU,
            [=](double x) { return (x > 0) ? x : alpha * std::expm1(x); },
            [=](double dy, double x, double y) { return dy * ((x > 0) ? 1 : y + alpha); });
        add(input,
            "input",
            get_input_tensor(tensor_elem_gen_integer{miopen_type<T>{} == miopenHalf ? 5 : 17}));
        add(alpha, "alpha");
        add(beta, "beta");
        add(gamma, "gamma");
        add(mode, "mode", generate_data(modes()));
        add(packed, "packed", generate_data({true, false}));
    }

    std::vector<std::string> modes()
    {
        std::vector<std::string> result(lookup.size());
        std::transform(lookup.begin(), lookup.end(), result.begin(), select_first{});
        return result;
    }

    miopen::ActivationDescriptor make_descriptor(miopenActivationMode_t m) const
    {
        return {m, alpha, beta, gamma};
    }

    static std::string transform_mode(std::string s)
    {
        return miopen::RemovePrefix(miopen::ToUpper(s), "MIOPENACTIVATION");
    }

    void run()
    {
        if(!packed)
        {
            const auto dim_lens = input.desc.GetLengths();
            auto dim_strides    = input.desc.GetStrides();
            dim_strides[0]      = dim_strides[0] + 1;

            input = tensor<T>{dim_lens, dim_strides};
        }

        std::size_t n, c, h, w;
        std::tie(n, c, h, w) = miopen::tien<4>(input.desc.GetLengths());
        size_t total_mem     = 4 * input.desc.GetNumBytes(); // estimate based on backward pass
        size_t device_mem    = get_handle().GetGlobalMemorySize();
        if(total_mem >= device_mem)
        {
            show_command();
            std::cout << "Config requires " << total_mem
                      << " Bytes to write all necessary tensors to GPU. GPU has " << device_mem
                      << " Bytes of memory." << std::endl;
            return;
        }

        lookup[transform_mode(mode)]();
    }

    template <class Forward, class Backward>
    void run(miopenActivationMode_t m, Forward f, Backward b)
    {
        auto desc = make_descriptor(m);
        auto out  = verify(verify_forward_activation<T>{input, desc}, f);
        auto dout = out.first;
        dout.generate([&](int n, int c, int h, int w) {
            T x      = out.first(n, c, h, w);
            double y = (877 * n + 547 * c + 701 * h + 1049 * w + static_cast<int>(769 * x)) % 2503;
            return ((x * y) / 1301.0);
        });
        verify(verify_backwards_activation<T>{input, dout, out.first, desc}, b);
    }
};

int main(int argc, const char* argv[]) { test_drive<activation_driver>(argc, argv); }
