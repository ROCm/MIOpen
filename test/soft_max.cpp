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

#define NEGATIVE_CUTOFF_VAL_FP32 (-1e20)
#define NEGATIVE_CUTOFF_VAL_FP16 (-1e4)

template <typename T>
T logaddexp(T x, T y, T neg_inf)
{
    T a = std::max(x, y);
    T b = std::min(x, y);
    T c = b - a;

    return c <= neg_inf ? std::max(a, neg_inf) : std::max(T(a + log(T(1) + exp(b - a))), neg_inf);
}

template <class T>
struct verify_forward_sofmax
{
    tensor<T> input;
    tensor<T> output;

    float alpha;
    float beta;
    miopenSoftmaxAlgorithm_t algo;
    miopenSoftmaxMode_t mode;

    verify_forward_sofmax(const tensor<T>& pinput,
                          const tensor<T>& pout,
                          float palpha                = 1,
                          float pbeta                 = 0,
                          miopenSoftmaxAlgorithm_t pa = MIOPEN_SOFTMAX_ACCURATE,
                          miopenSoftmaxMode_t pm      = MIOPEN_SOFTMAX_MODE_CHANNEL)
    {
        input  = pinput;
        output = pout;
        alpha  = palpha;
        beta   = pbeta;
        algo   = pa;
        mode   = pm;
    }

    tensor<T> cpu() const
    {
        auto out = output;

        int in_n, in_c, in_h, in_w;
        std::tie(in_n, in_c, in_h, in_w) = miopen::tien<4>(input.desc.GetLengths());

        int in_nstr, in_cstr, in_hstr;
        std::tie(in_nstr, in_cstr, in_hstr, std::ignore) = miopen::tien<4>(input.desc.GetStrides());

        int out_nstr, out_cstr, out_hstr;
        std::tie(out_nstr, out_cstr, out_hstr, std::ignore) =
            miopen::tien<4>(out.desc.GetStrides());

        if(mode == MIOPEN_SOFTMAX_MODE_INSTANCE)
            par_ford(in_n)([&](int o) {
                if(algo == MIOPEN_SOFTMAX_FAST)
                {
                    double sum = 0;
                    ford(in_c, in_h, in_w)([&](int w, int i, int j) {
                        sum += std::exp(input[o * in_nstr + w * in_cstr + i * in_hstr + j]);
                    });
                    ford(in_c, in_h, in_w)([&](int w, int i, int j) {
                        out[o * out_nstr + w * out_cstr + i * out_hstr + j] =
                            alpha * (std::exp(input[o * in_nstr + w * in_cstr + i * in_hstr + j]) /
                                     sum) +
                            beta * out[o * out_nstr + w * out_cstr + i * out_hstr + j];
                    });
                }
                else
                {
                    T max_c = std::numeric_limits<T>::lowest();
                    ford(in_c, in_h, in_w)([&](int w, int i, int j) {
                        max_c = std::max(max_c, input[o * in_nstr + w * in_cstr + i * in_hstr + j]);
                    });

                    if(algo == MIOPEN_SOFTMAX_LOG)
                    {
                        double neg_inf = input.desc.GetType() == miopenHalf
                                             ? NEGATIVE_CUTOFF_VAL_FP16
                                             : NEGATIVE_CUTOFF_VAL_FP32;
                        double sum = neg_inf;
                        ford(in_c, in_h, in_w)([&](int w, int i, int j) {
                            sum = logaddexp(
                                double(input[o * in_nstr + w * in_cstr + i * in_hstr + j] - max_c),
                                sum,
                                neg_inf);
                        });

                        ford(in_c, in_h, in_w)([&](int w, int i, int j) {
                            out[o * out_nstr + w * out_cstr + i * out_hstr + j] =
                                alpha * (input[o * in_nstr + w * in_cstr + i * in_hstr + j] -
                                         max_c - sum) +
                                beta * out[o * out_nstr + w * out_cstr + i * out_hstr + j];
                        });
                    }
                    else
                    {
                        double sum = 0;
                        ford(in_c, in_h, in_w)([&](int w, int i, int j) {
                            sum += std::exp(input[o * in_nstr + w * in_cstr + i * in_hstr + j] -
                                            max_c);
                        });

                        ford(in_c, in_h, in_w)([&](int w, int i, int j) {
                            out[o * out_nstr + w * out_cstr + i * out_hstr + j] =
                                alpha *
                                    (std::exp(input[o * in_nstr + w * in_cstr + i * in_hstr + j] -
                                              max_c) /
                                     sum) +
                                beta * out[o * out_nstr + w * out_cstr + i * out_hstr + j];
                        });
                    }
                }
            });
        else
            par_ford(in_n, in_h, in_w)([&](int o, int i, int j) {
                if(algo == MIOPEN_SOFTMAX_FAST)
                {
                    double sum = 0;
                    ford(in_c)([&](int w) {
                        sum += std::exp(input[o * in_nstr + w * in_cstr + i * in_hstr + j]);
                    });
                    ford(in_c)([&](int w) {
                        out[o * out_nstr + w * out_cstr + i * out_hstr + j] =
                            alpha * (std::exp(input[o * in_nstr + w * in_cstr + i * in_hstr + j]) /
                                     sum) +
                            beta * out[o * out_nstr + w * out_cstr + i * out_hstr + j];
                    });
                }
                else
                {
                    T max_c = std::numeric_limits<T>::lowest();
                    ford(in_c)([&](int w) {
                        max_c = std::max(max_c, input[o * in_nstr + w * in_cstr + i * in_hstr + j]);
                    });

                    if(algo == MIOPEN_SOFTMAX_LOG)
                    {
                        double neg_inf = input.desc.GetType() == miopenHalf
                                             ? NEGATIVE_CUTOFF_VAL_FP16
                                             : NEGATIVE_CUTOFF_VAL_FP32;
                        double sum = neg_inf;
                        ford(in_c)([&](int w) {
                            sum = logaddexp(
                                double(input[o * in_nstr + w * in_cstr + i * in_hstr + j] - max_c),
                                sum,
                                neg_inf);
                        });

                        ford(in_c)([&](int w) {
                            out[o * out_nstr + w * out_cstr + i * out_hstr + j] =
                                alpha * (input[o * in_nstr + w * in_cstr + i * in_hstr + j] -
                                         max_c - sum) +
                                beta * out[o * out_nstr + w * out_cstr + i * out_hstr + j];
                        });
                    }
                    else
                    {
                        double sum = 0;
                        ford(in_c)([&](int w) {
                            sum += std::exp(input[o * in_nstr + w * in_cstr + i * in_hstr + j] -
                                            max_c);
                        });

                        ford(in_c)([&](int w) {
                            out[o * out_nstr + w * out_cstr + i * out_hstr + j] =
                                alpha *
                                    (std::exp(input[o * in_nstr + w * in_cstr + i * in_hstr + j] -
                                              max_c) /
                                     sum) +
                                beta * out[o * out_nstr + w * out_cstr + i * out_hstr + j];
                        });
                    }
                }
            });
        return out;
    }

    tensor<T> gpu() const
    {
        auto&& handle = get_handle();
        auto out      = output;

        auto in_dev  = handle.Write(input.data);
        auto out_dev = handle.Write(out.data);

        miopen::SoftmaxForward(
            handle, &alpha, &beta, input.desc, in_dev.get(), out.desc, out_dev.get(), algo, mode);

        out.data = handle.Read<T>(out_dev, out.data.size());
        return out;
    }

    void fail(int = 0) const
    {
        std::cout << "Forward Sofmax: " << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
    }
};

template <class T>
struct verify_backward_sofmax
{
    tensor<T> dinput;
    tensor<T> dout;
    tensor<T> out;

    float alpha;
    float beta;
    miopenSoftmaxAlgorithm_t algo;
    miopenSoftmaxMode_t mode;

    verify_backward_sofmax(const tensor<T>& pout,
                           const tensor<T>& pdout,
                           const tensor<T>& pdinput,
                           float palpha                = 1,
                           float pbeta                 = 0,
                           miopenSoftmaxAlgorithm_t pa = MIOPEN_SOFTMAX_ACCURATE,
                           miopenSoftmaxMode_t pm      = MIOPEN_SOFTMAX_MODE_CHANNEL)
    {
        dinput = pdinput;
        dout   = pdout;
        out    = pout;
        alpha  = palpha;
        beta   = pbeta;
        algo   = pa;
        mode   = pm;
    }

    tensor<T> cpu() const
    {
        auto din = dinput;

        int in_n, in_c, in_h, in_w;
        std::tie(in_n, in_c, in_h, in_w) = miopen::tien<4>(din.desc.GetLengths());

        int in_nstr, in_cstr, in_hstr;
        std::tie(in_nstr, in_cstr, in_hstr, std::ignore) = miopen::tien<4>(din.desc.GetStrides());

        int out_nstr, out_cstr, out_hstr;
        std::tie(out_nstr, out_cstr, out_hstr, std::ignore) =
            miopen::tien<4>(dout.desc.GetStrides());

        if(mode == MIOPEN_SOFTMAX_MODE_INSTANCE)
            par_ford(in_n)([&](int o) {
                double sum = 0;
                ford(in_c, in_h, in_w)([&](int c, int i, int j) {
                    if(algo == MIOPEN_SOFTMAX_LOG)
                        sum += dout[o * out_nstr + c * out_cstr + i * out_hstr + j];
                    else
                        sum += out[o * out_nstr + c * out_cstr + i * out_hstr + j] *
                               dout[o * out_nstr + c * out_cstr + i * out_hstr + j];
                });

                ford(in_c, in_h, in_w)([&](int c, int i, int j) {
                    if(algo == MIOPEN_SOFTMAX_LOG)
                        din[o * in_nstr + c * in_cstr + i * in_hstr + j] =
                            T(alpha *
                                  (dout[o * out_nstr + c * out_cstr + i * out_hstr + j] -
                                   sum * std::exp(
                                             out[o * out_nstr + c * out_cstr + i * out_hstr + j])) +
                              beta * din[o * in_nstr + c * in_cstr + i * in_hstr + j]);
                    else
                        din[o * in_nstr + c * in_cstr + i * in_hstr + j] =
                            alpha * (out[o * out_nstr + c * out_cstr + i * out_hstr + j] *
                                     (dout[o * out_nstr + c * out_cstr + i * out_hstr + j] - sum)) +
                            beta * din[o * in_nstr + c * in_cstr + i * in_hstr + j];
                });
            });
        else
            par_ford(in_n, in_h, in_w)([&](int o, int i, int j) {
                double sum = 0;
                ford(in_c)([&](int c) {
                    if(algo == MIOPEN_SOFTMAX_LOG)
                        sum += dout[o * out_nstr + c * out_cstr + i * out_hstr + j];
                    else
                        sum += out[o * out_nstr + c * out_cstr + i * out_hstr + j] *
                               dout[o * out_nstr + c * out_cstr + i * out_hstr + j];
                });

                ford(in_c)([&](int c) {
                    if(algo == MIOPEN_SOFTMAX_LOG)
                        din[o * in_nstr + c * in_cstr + i * in_hstr + j] =
                            alpha *
                                (dout[o * out_nstr + c * out_cstr + i * out_hstr + j] -
                                 sum * std::exp(
                                           out[o * out_nstr + c * out_cstr + i * out_hstr + j])) +
                            beta * din[o * in_nstr + c * in_cstr + i * in_hstr + j];
                    else
                        din[o * in_nstr + c * in_cstr + i * in_hstr + j] =
                            alpha * (out[o * out_nstr + c * out_cstr + i * out_hstr + j] *
                                     (dout[o * out_nstr + c * out_cstr + i * out_hstr + j] - sum)) +
                            beta * din[o * in_nstr + c * in_cstr + i * in_hstr + j];
                });
            });
        return din;
    }

    tensor<T> gpu() const
    {
        auto&& handle = get_handle();
        auto din      = dinput;

        auto din_dev  = handle.Write(din.data);
        auto dout_dev = handle.Write(dout.data);
        auto out_dev  = handle.Write(out.data);

        miopen::SoftmaxBackward(handle,
                                &alpha,
                                out.desc,
                                out_dev.get(),
                                dout.desc,
                                dout_dev.get(),
                                &beta,
                                din.desc,
                                din_dev.get(),
                                algo,
                                mode);

        din.data = handle.Read<T>(din_dev, din.data.size());
        return din;
    }

    void fail(int = 0) const
    {
        std::cout << "Backward Sofmax: " << std::endl;
        std::cout << "Output tensor: " << out.desc.ToString() << std::endl;
    }
};

template <class T>
struct softmax_driver : test_driver
{
    tensor<T> input;
    tensor<T> out;
    tensor<T> din;
    tensor<T> dout;

    std::vector<int> in_dim;
    std::vector<float> scales;
    int algo_cmd = 1;
    int mode_cmd = 1;

    softmax_driver()
    {
        std::set<std::vector<int>> in_dim_set = get_inputs(batch_factor);

        /// \todo Resolve this workaround. Random failure on Jenkins (ROCm3.0):
        /// --float --input-dim 1 480 128 256 --algorithm 2 --mode 1 --scales 1 0 --tolerance 8000
        /// FAILED: inf
        in_dim_set.erase({1, 480, 128, 256});

        /// \todo Resolve this workaround. Regular failures on Radeon VII, ROCm 3.3:
        /// --float --input-dim 1 1 8 8 --algorithm 0 --mode 1 --scales 1 0 --tolerance 8000
        /// FAILED: -nan
        in_dim_set.erase({1, 1, 8, 8});
        in_dim_set.erase({1, 1, 14, 14});
        in_dim_set.erase({1, 1, 27, 27});
        in_dim_set.erase({1, 32, 7, 7});
        in_dim_set.erase({1, 32, 8, 8});

        std::vector<std::vector<int>> in_dim_vec(in_dim_set.begin(), in_dim_set.end());

        add(in_dim, "input-dim", generate_data(in_dim_vec, {16, 32, 8, 8}));

        add(algo_cmd, "algorithm", generate_data({0, 1, 2}));
        add(mode_cmd, "mode", generate_data({0, 1}));

        add(scales, "scales", generate_data({{1.f, 0.f}, {float(0.5), float(0.5)}}));
        add(tolerance, "tolerance", generate_data({8000})); // 80 for MIOPEN_SOFTMAX_MODE_CHANNEL
    }

    void run()
    {
        miopenSoftmaxAlgorithm_t algo = miopenSoftmaxAlgorithm_t(algo_cmd);
        miopenSoftmaxMode_t mode      = miopenSoftmaxMode_t(mode_cmd);
        unsigned long max_value =
            miopen_type<T>{} == miopenHalf ? (algo == MIOPEN_SOFTMAX_LOG ? 3 : 5) : 17;

        /// \todo Apply mix-precision in softmax to improve the stability of fp16
        if((in_dim[1] * in_dim[2] * in_dim[3] >= 2048) && mode == MIOPEN_SOFTMAX_MODE_INSTANCE &&
           miopen_type<T>{} == miopenHalf)
            return;
        if(in_dim[1] >= 96 && in_dim[2] >= 14 && in_dim[3] >= 14 && algo == MIOPEN_SOFTMAX_FAST &&
           miopen_type<T>{} == miopenHalf)
            return;

        input             = tensor<T>{in_dim}.generate(tensor_elem_gen_integer{max_value});
        size_t total_mem  = 2 * input.desc.GetNumBytes(); // estimate based on backward pass
        size_t device_mem = get_handle().GetGlobalMemorySize();
        if(total_mem >= device_mem)
        {
            show_command();
            std::cout << "Config requires " << total_mem
                      << " Bytes to write all necessary tensors to GPU. GPU has " << device_mem
                      << " Bytes of memory." << std::endl;
            return;
        }

        out         = tensor<T>{in_dim}.generate(tensor_elem_gen_integer{max_value});
        float alpha = scales[0];
        float beta  = scales[1];

        verify(verify_forward_sofmax<T>{input, out, alpha, beta, algo, mode});
        dout = tensor<T>{in_dim}.generate([&](int n, int c, int h, int w) {
            T x      = input(n, c, h, w);
            double y = (877 * n + 547 * c + 701 * h + 1049 * w + static_cast<int>(769 * x)) % 2503;
            return ((x * y) / 1301.0);
        });
        din  = tensor<T>{in_dim}.generate(tensor_elem_gen_integer{max_value});
        verify(verify_backward_sofmax<T>{out, dout, din, alpha, beta, algo, mode});
    }
};

int main(int argc, const char* argv[]) { test_drive<softmax_driver>(argc, argv); }
