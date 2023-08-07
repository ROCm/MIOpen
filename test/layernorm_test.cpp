/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include <miopen/layernorm.hpp>
#include <miopen/tensor.hpp>
#include <utility>

#include "driver.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

template <class T>
struct verify_forward_layernorm
{
    tensor<T> input;
    tensor<T> weight;
    tensor<T> bias;
    tensor<T> output;
    tensor<T> mean;
    tensor<T> rstd;
    double eps;
    int dim;
    miopenLayerNormMode_t mode;

    verify_forward_layernorm(const tensor<T>& pinput,
                             const tensor<T>& pweight,
                             const tensor<T>& pbias,
                             tensor<T>& pout,
                             tensor<T>& pmean,
                             tensor<T>& prstd,
                             double peps,
                             int pdim,
                             miopenLayerNormMode_t pm)
    {
        input  = pinput;
        weight = pweight;
        bias   = pbias;
        output = pout;
        mean   = pmean;
        rstd   = prstd;
        eps    = peps;
        dim    = pdim;
        mode   = pm;
    }

    std::tuple<tensor<T>, tensor<T>, tensor<T>> cpu() const
    {
        auto dims         = input.desc.GetLengths();
        size_t grid_size  = 1;
        size_t outer_size = 1;
        size_t inner_size = 1;
        size_t i          = 0;
        for(; i < dim; i++)
        {
            outer_size *= dims[i];
            grid_size *= dims[i];
        }

        for(; i < dims.size(); i++)
        {
            inner_size *= dims[i];
            grid_size *= dims[i];
        }

        auto toutput = output;
        auto tmean   = mean;
        auto trstd   = rstd;

        par_ford(outer_size)([&](int o) {
            double mean_v = 0;
            double var_v  = 0;

            ford(inner_size)([&](int i) {
                float tmp = input[o * inner_size + i];
                mean_v += tmp;
                mean_v += tmp * tmp;
            });

            mean_v /= inner_size;
            var_v /= inner_size - mean_v * mean_v;

            tmean[o] = mean_v;
            trstd[o] = sqrt(var_v + eps);

            ford(inner_size)([&](int i) {
                double weight_v = (weight.data.size() == 0) ? weight[i] : 1;
                double bias_v   = (bias.data.size() == 0) ? bias[i] : 0;
                toutput[o * inner_size + i] =
                    (input[o * inner_size + i] - mean_v) * sqrt(var_v + eps) * weight_v + bias_v;
            });
        });
        return std::make_tuple(toutput, tmean, trstd);
    }

    std::tuple<tensor<T>, tensor<T>, tensor<T>> gpu() const
    {
        auto&& handle = get_handle();

        auto toutput = output;
        auto tmean   = mean;
        auto trstd   = rstd;

        auto in_dev     = handle.Write(input.data);
        auto weight_dev = handle.Write(weight.data);
        auto bias_dev   = handle.Write(bias.data);
        auto out_dev    = handle.Write(toutput.data);
        auto mean_dev   = handle.Write(tmean.data);
        auto rstd_dev   = handle.Write(trstd.data);

        miopen::LayerNormForward(handle,
                                 input.desc,
                                 in_dev.get(),
                                 weight.desc,
                                 weight_dev.get(),
                                 bias.desc,
                                 bias_dev.get(),
                                 toutput.desc,
                                 out_dev.get(),
                                 tmean.desc,
                                 mean_dev.get(),
                                 trstd.desc,
                                 rstd_dev.get(),
                                 mode,
                                 eps,
                                 dim);

        toutput.data = handle.Read<T>(out_dev, toutput.data.size());
        tmean.data   = handle.Read<T>(mean_dev, tmean.data.size());
        trstd.data   = handle.Read<T>(rstd_dev, trstd.data.size());

        return std::make_tuple(toutput, tmean, trstd);
    }

    void fail(int = 0) const
    {
        std::cout << "Forward LayerNorm: " << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
    }
};

template <class T>
struct verify_backward_layernorm
{
    tensor<T> input;
    tensor<T> doutput;
    tensor<T> weight;
    tensor<T> mean;
    tensor<T> rstd;
    tensor<T> dinput;
    tensor<T> dweight;
    tensor<T> dbias;
    int dim;
    miopenLayerNormMode_t mode;

    verify_backward_layernorm(const tensor<T>& pinput,
                              const tensor<T>& pdoutput,
                              const tensor<T>& pweight,
                              const tensor<T>& pmean,
                              const tensor<T>& prstd,
                              tensor<T>& pdinput,
                              tensor<T>& pdweight,
                              tensor<T>& pdbias,
                              int pdim,
                              miopenLayerNormMode_t pm)
    {
        input   = pinput;
        doutput = pdoutput;
        weight  = pweight;
        mean    = pmean;
        rstd    = prstd;
        dinput  = pdinput;
        dweight = pdweight;
        dbias   = pdbias;
        dim     = pdim;
        mode    = pm;
    }

    std::tuple<tensor<T>, tensor<T>, tensor<T>> cpu() const
    {
        auto dims         = input.desc.GetLengths();
        size_t grid_size  = 1;
        size_t outer_size = 1;
        size_t inner_size = 1;
        size_t i          = 0;
        for(; i < dim; i++)
        {
            outer_size *= dims[i];
            grid_size *= dims[i];
        }

        for(; i < dims.size(); i++)
        {
            inner_size *= dims[i];
            grid_size *= dims[i];
        }

        auto tdinput = dinput;

        par_ford(outer_size)([&](int o) {
            double sum1 = 0;
            double sum2 = 0;
            ford(inner_size)([&](int i) {
                double weight_v = (weight.data.size() == 0) ? weight[o * inner_size + i] : 1;
                double dy       = (doutput.data.size() == 0) ? doutput[o * inner_size + i] : 0;
                double x        = input[i * inner_size + o];

                sum1 += dy * x * weight_v;
                sum2 += dy * weight_v;
            });

            double s = 1.0 / inner_size;

            double mean_v = mean[o];
            double rstd_v = rstd[o];

            double a  = (sum2 * mean_v - sum1) * rstd_v * rstd_v * rstd_v * s;
            double c2 = -(a * mean_v + sum2 * rstd_v * s);

            ford(inner_size)([&](int i) {
                double weight_v = (weight.data.size() == 0) ? weight[o * inner_size + i] : 1;
                double dy       = (doutput.data.size() == 0) ? doutput[o * inner_size + i] : 0;
                double x        = input[i * inner_size + o];

                double val                  = rstd_v * dy * weight_v + a * x + c2;
                tdinput[i * inner_size + o] = val;
            });
        });

        auto tdweight = dweight;
        auto tdbias   = dbias;
        if((dweight.data.size() != 0) || dbias.data.size() != 0)
        {
            par_ford(inner_size)([&](int i) {
                double sum1 = 0;
                double sum2 = 0;

                ford(outer_size)([&](int o) {
                    double dy = (doutput.data.size() == 0) ? doutput[i * inner_size + o] : 0;
                    double x  = input[i * inner_size + o];

                    sum1 += dy * (x - mean[o]) * rstd[o];
                    sum2 += dy;
                });

                tdweight[i] = sum1;
                tdbias[i]   = sum2;
            });
        }

        return std::make_tuple(tdinput, tdweight, tdbias);
    }

    std::tuple<tensor<T>, tensor<T>, tensor<T>> gpu() const
    {
        auto&& handle = get_handle();

        auto tdinput  = dinput;
        auto tdweight = dweight;
        auto tdbias   = dbias;

        auto in_dev     = handle.Write(input.data);
        auto dout_dev   = handle.Write(doutput.data);
        auto weight_dev = handle.Write(weight.data);
        auto mean_dev   = handle.Write(mean.data);
        auto rstd_dev   = handle.Write(rstd.data);
        auto din_dev    = handle.Write(tdinput.data);
        auto dw_dev     = handle.Write(tdweight.data);
        auto db_dev     = handle.Write(tdbias.data);

        miopen::LayerNormBackward(handle,
                                  input.desc,
                                  in_dev.get(),
                                  doutput.desc,
                                  dout_dev.get(),
                                  weight.desc,
                                  weight_dev.get(),
                                  mean.desc,
                                  mean_dev.get(),
                                  rstd.desc,
                                  rstd_dev.get(),
                                  tdinput.desc,
                                  din_dev.get(),
                                  tdweight.desc,
                                  dw_dev.get(),
                                  tdbias.desc,
                                  db_dev.get(),
                                  mode,
                                  dim);

        tdinput.data  = handle.Read<T>(din_dev, tdinput.data.size());
        tdweight.data = handle.Read<T>(dw_dev, tdweight.data.size());
        tdbias.data   = handle.Read<T>(db_dev, tdbias.data.size());

        return std::make_tuple(tdinput, tdweight, tdbias);
    }

    void fail(int = 0) const
    {
        std::cout << "Backward LayerNorm: " << std::endl;
        std::cout << "DInput tensor: " << dinput.desc.ToString() << std::endl;
    }
};

template <class T>
struct layernorm_driver : test_driver
{
    tensor<T> input;
    tensor<T> weight;
    tensor<T> bias;
    tensor<T> output;
    tensor<T> mean;
    tensor<T> rstd;
    tensor<T> doutput;
    tensor<T> dinput;
    tensor<T> dweight;
    tensor<T> dbias;

    double eps_cmd;
    int dim_cmd;
    int mode_cmd;

    std::vector<int> in_dim;

    layernorm_driver()
    {
        std::set<std::vector<int>> in_dim_set = get_ln_inputs(batch_factor);

        std::vector<std::vector<int>> in_dim_vec(in_dim_set.begin(), in_dim_set.end());

        add(in_dim, "input-dim", generate_data(in_dim_vec, {16, 32, 8, 8, 8}));

        add(mode_cmd, "mode", generate_data({0, 1}));

        add(dim_cmd, "dim", generate_data({0, 1, 2, 3, 4}));

        add(eps_cmd, "eps", generate_data({1e-5}));
    }

    void run()
    {
        miopenLayerNormMode_t mode = miopenLayerNormMode_t(mode_cmd);
        unsigned long max_value;
        if((miopen_type<T>{} == miopenHalf) || miopen_type<T>{} == miopenBFloat16)
            max_value = 5;
        else
            max_value = 17;

        input = tensor<T>{in_dim}.generate(tensor_elem_gen_integer{max_value});

        if(mode == MIOPEN_ELEMENTWISE_AFFINE)
        {
            std::vector<int> inner_dim;
            if(dim_cmd == in_dim.size())
                inner_dim = {1};
            else
                inner_dim = {in_dim.begin() + dim_cmd, in_dim.end()};
            weight = tensor<T>{inner_dim}.generate(tensor_elem_gen_integer{max_value});
            bias   = tensor<T>{inner_dim}.generate(tensor_elem_gen_integer{max_value});
        }

        std::vector<int> outer_dim;
        if(dim_cmd == 0)
            outer_dim = {1};
        else
            outer_dim = {in_dim.begin(), in_dim.end() - (in_dim.size() - dim_cmd)};

        mean = tensor<T>{outer_dim}.generate(tensor_elem_gen_integer{max_value});
        rstd = tensor<T>{outer_dim}.generate(tensor_elem_gen_integer{max_value});

        size_t total_mem =
            2 * (input.desc.GetNumBytes() + weight.desc.GetNumBytes() + bias.desc.GetNumBytes() +
                 mean.desc.GetNumBytes() + rstd.desc.GetNumBytes());
        size_t device_mem = get_handle().GetGlobalMemorySize();
        if(total_mem >= device_mem)
        {
            show_command();
            std::cout << "Config requires " << total_mem
                      << " Bytes to write all necessary tensors to GPU. GPU has " << device_mem
                      << " Bytes of memory." << std::endl;
            return;
        }

        output     = tensor<T>{in_dim}.generate(tensor_elem_gen_integer{max_value});
        double eps = eps_cmd;
        int dim    = dim_cmd;
        verify(
            verify_forward_layernorm<T>{input, weight, bias, output, mean, rstd, eps, dim, mode});

        doutput = tensor<T>{in_dim}.generate(tensor_elem_gen_integer{max_value});
        dinput  = tensor<T>{in_dim}.generate(tensor_elem_gen_integer{max_value});

        if(mode == MIOPEN_ELEMENTWISE_AFFINE)
        {
            std::vector<int> inner_dim;
            if(dim_cmd == in_dim.size())
                inner_dim = {1};
            else
                inner_dim = {in_dim.begin() + dim_cmd, in_dim.end()};
            dweight = tensor<T>{inner_dim}.generate(tensor_elem_gen_integer{max_value});
            dbias   = tensor<T>{inner_dim}.generate(tensor_elem_gen_integer{max_value});
        }

        verify(verify_backward_layernorm<T>{
            input, doutput, weight, mean, rstd, dinput, dweight, dbias, dim, mode});
    }
};

int main(int argc, const char* argv[]) { test_drive<layernorm_driver>(argc, argv); }
