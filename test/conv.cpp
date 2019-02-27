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
#include <miopen/convolution.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/solver.hpp>
#include <utility>

#include "driver.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <miopen/stringutils.hpp>
#include "tensor_util.hpp"

#define TEST_DIRECT_SUPPORTED_CONFIG_ONLY (!MIOPEN_USE_ROCBLAS)

#if TEST_DIRECT_SUPPORTED_CONFIG_ONLY
static bool is_direct_fwd_bwd_data_supported(miopen::Handle&,
                                             const miopen::ConvolutionDescriptor convDesc,
                                             const miopen::TensorDescriptor&,
                                             const miopen::TensorDescriptor& wDesc,
                                             const miopen::TensorDescriptor&)
{
    return convDesc.IsDirectSupported(wDesc) &&
           (convDesc.GetConvDilations()[0] == 1 && convDesc.GetConvDilations()[1] == 1);
}

static bool is_direct_bwd_wrw_supported(miopen::Handle& handle,
                                        const miopen::ConvolutionDescriptor convDesc,
                                        const miopen::TensorDescriptor& xDesc,
                                        const miopen::TensorDescriptor& wDesc,
                                        const miopen::TensorDescriptor& yDesc)
{
    mlo_construct_BwdWrW2D construct_params(xDesc, wDesc, yDesc, convDesc, 0);
    construct_params.setDoSearch(false);
    construct_params.saveSearchRequest(false);
    construct_params.setGeneralCompOptions("");
    construct_params.setStream(&handle);

    return !FindAllSolutions(construct_params).empty();
}
#endif

static bool is_int8_workspace_valid(miopen::Handle& handle,
                                    const miopen::ConvolutionDescriptor convDesc,
                                    const miopen::TensorDescriptor& xDesc,
                                    const miopen::TensorDescriptor& wDesc,
                                    const miopen::TensorDescriptor& yDesc)
{
    return !(((wDesc.GetLengths().at(2) == 1 && wDesc.GetLengths().at(3) == 1 &&
               convDesc.GetConvPads()[0] == 0 && convDesc.GetConvPads()[1] == 0) &&
              ((xDesc.GetLengths().at(2) <= 14 && xDesc.GetLengths().at(3) <= 14 &&
                convDesc.GetConvStrides()[0] == 1 && convDesc.GetConvStrides()[1] == 1) ||
               (convDesc.GetConvStrides()[0] == 2 && convDesc.GetConvStrides()[1] == 2)) &&
              (convDesc.ForwardGetWorkSpaceSize(handle, wDesc, xDesc, yDesc) <
               convDesc.ForwardGetWorkSpaceSizeGEMMTranspose(xDesc, yDesc))) ||
             (convDesc.ForwardGetWorkSpaceSize(handle, wDesc, xDesc, yDesc) <
              convDesc.ForwardGetWorkSpaceSizeGEMM(wDesc, yDesc)));
}

struct scalar_gen_random_float
{
    double min_val = 0;
    double max_val = 1;

    double operator()() const
    {
        return min_val + (max_val - min_val) * double(std::rand()) / RAND_MAX;
    }
};

struct scalar_gen_random_integer
{
    unsigned long min_val = 1;
    unsigned long max_val = 16;

    double operator()() const { return min_val + std::rand() % (max_val - min_val + 1); }
};

struct tensor_elem_gen_one
{
    template <class... Ts>
    double operator()(Ts...) const
    {
        return 1;
    }
};

template <class T>
tensor<T> get_output_tensor(const miopen::ConvolutionDescriptor& filter,
                            const tensor<T>& input,
                            const tensor<T>& weights)
{
    return tensor<T>{filter.GetForwardOutputTensor(input.desc, weights.desc)};
}

template <class T>
tensor<float> get_output_tensor_int8(const miopen::ConvolutionDescriptor& filter,
                                     const tensor<T>& input,
                                     const tensor<T>& weights)
{
    return tensor<float>{filter.GetForwardOutputTensor(input.desc, weights.desc)};
}

template <class T>
struct conv_base
{
    tensor<T> input;
    tensor<T> weights;
    tensor<T> out;
    tensor<float> out_int8;
    miopen::ConvolutionDescriptor filter;
    int bias{};
    int search{};

    void fail(float = 0) const
    {
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
        std::cout << "Weights tensor: " << weights.desc.ToString() << std::endl;
        std::cout << "Output tensor: " << out.desc.ToString() << std::endl;
        std::cout << "Filter: " << filter << std::endl;
    }
};

template <class T>
struct verify_forward_conv : conv_base<T>
{
    using conv_base<T>::out;
    using conv_base<T>::input;
    using conv_base<T>::weights;
    using conv_base<T>::filter;
    using conv_base<T>::bias;
    using conv_base<T>::search;

    verify_forward_conv(const tensor<T>& pinput,
                        const tensor<T>& pweights,
                        const miopen::ConvolutionDescriptor& pfilter,
                        int pbias   = 0,
                        int psearch = 0)
    {
        input   = pinput;
        weights = pweights;
        filter  = pfilter;
        bias    = pbias;
        search  = psearch;
    }

    tensor<T> cpu() const
    {
        auto rout = get_output_tensor(filter, input, weights);

        if(filter.mode == miopenTranspose)
        {
            std::fill(rout.begin(), rout.end(), 0);

            int in_n, in_c, in_h, in_w;
            std::tie(in_n, in_c, in_h, in_w) = miopen::tien<4>(input.desc.GetLengths());

            int wei_k, wei_c, wei_h, wei_w;
            std::tie(wei_k, wei_c, wei_h, wei_w) = miopen::tien<4>(weights.desc.GetLengths());

            int out_c, out_h, out_w;
            std::tie(std::ignore, out_c, out_h, out_w) = miopen::tien<4>(rout.desc.GetLengths());

            par_ford(in_n, out_c, out_h, out_w)([&](int n, int c, int hi, int wi) {
                const int group_id = c / wei_c;
                double acc         = 0;
                ford(wei_k / filter.group_count, wei_h, wei_w)([&](int k, int x, int y) {
                    int h_ = filter.GetConvPads()[0] + hi - x * filter.GetConvDilations()[0];
                    int w_ = filter.GetConvPads()[1] + wi - y * filter.GetConvDilations()[1];

                    int ho = h_ / filter.GetConvStrides()[0];
                    int wo = w_ / filter.GetConvStrides()[1];

                    if(((ho * filter.GetConvStrides()[0] == h_) and
                        (wo * filter.GetConvStrides()[1] == w_)) and
                       ((ho >= 0 and ho < in_h) and (wo >= 0 and wo < in_w)))
                    {
                        const int in_ch  = group_id * (wei_k / filter.group_count) + k;
                        const int wei_ch = c % wei_c;
                        acc +=
                            double(input(n, in_ch, ho, wo)) * double(weights(in_ch, wei_ch, x, y));
                    }
                });
                rout(n, c, hi, wi) = acc;
            });
        }
        else
        {
            int in_h, in_w;
            std::tie(std::ignore, std::ignore, in_h, in_w) =
                miopen::tien<4>(input.desc.GetLengths());

            int wei_k, wei_c, wei_h, wei_w;
            std::tie(wei_k, wei_c, wei_h, wei_w) = miopen::tien<4>(weights.desc.GetLengths());

            rout.par_for_each([&](int n, int k, int ho, int wo) {
                const int start_hi = ho * filter.GetConvStrides()[0] - filter.GetConvPads()[0];
                const int start_wi = wo * filter.GetConvStrides()[1] - filter.GetConvPads()[1];
                const int group_id = k / (wei_k / filter.group_count);

                double acc = bias;
                ford(wei_c, wei_h, wei_w)([&](int c_grp, int x, int y) {
                    const int hi = start_hi + x * filter.GetConvDilations()[0];
                    const int wi = start_wi + y * filter.GetConvDilations()[1];
                    const int c  = group_id * wei_c + c_grp;
                    if(hi >= 0 && hi < in_h && wi >= 0 && wi < in_w)
                    {
                        acc += double(input(n, c, hi, wi)) * double(weights(k, c_grp, x, y));
                    }
                });
                rout(n, k, ho, wo) = acc;
            });
        }

        return rout;
    }

    tensor<T> gpu() const
    {
        auto&& handle = get_handle();
        auto rout     = get_output_tensor(filter, input, weights);

        auto in_dev  = handle.Write(input.data);
        auto wei_dev = handle.Write(weights.data);
        auto out_dev = handle.Write(rout.data);

        size_t workspace_size =
            filter.mode == miopenTranspose
                ? filter.BackwardDataGetWorkSpaceSize(handle, weights.desc, input.desc, rout.desc)
                : filter.ForwardGetWorkSpaceSize(handle, weights.desc, input.desc, rout.desc);

        std::vector<char> workspace(workspace_size);
        auto workspace_dev = workspace_size != 0 ? handle.Write(workspace) : nullptr;

        int ret_algo_count;
        miopenConvAlgoPerf_t perf;

        float alpha = 1, beta = 0;

        if(filter.mode == miopenTranspose)
        {
            filter.FindConvBwdDataAlgorithm(handle,
                                            input.desc,
                                            in_dev.get(),
                                            weights.desc,
                                            wei_dev.get(),
                                            rout.desc,
                                            out_dev.get(),
                                            1,
                                            &ret_algo_count,
                                            &perf,
                                            workspace_dev.get(),
                                            workspace_size,
                                            search);

            filter.ConvolutionBackwardData(handle,
                                           &alpha,
                                           input.desc,
                                           in_dev.get(),
                                           weights.desc,
                                           wei_dev.get(),
                                           perf.bwd_data_algo,
                                           &beta,
                                           rout.desc,
                                           out_dev.get(),
                                           workspace_dev.get(),
                                           workspace_size);
        }
        else
        {
            filter.FindConvFwdAlgorithm(handle,
                                        input.desc,
                                        in_dev.get(),
                                        weights.desc,
                                        wei_dev.get(),
                                        rout.desc,
                                        out_dev.get(),
                                        1,
                                        &ret_algo_count,
                                        &perf,
                                        workspace_dev.get(),
                                        workspace_size,
                                        search);

            filter.ConvolutionForward(handle,
                                      &alpha,
                                      input.desc,
                                      in_dev.get(),
                                      weights.desc,
                                      wei_dev.get(),
                                      perf.fwd_algo,
                                      &beta,
                                      rout.desc,
                                      out_dev.get(),
                                      workspace_dev.get(),
                                      workspace_size);
        }

        rout.data = handle.Read<T>(out_dev, rout.data.size());

        return rout;
    }

    void fail(float = 0) const
    {
        std::cout << "Forward convolution: " << std::endl;
        this->conv_base<T>::fail();
    }
};

template <class T>
struct verify_forward_conv_int8 : conv_base<T>
{
    using conv_base<T>::out_int8;
    using conv_base<T>::input;
    using conv_base<T>::weights;
    using conv_base<T>::filter;
    using conv_base<T>::bias;
    using conv_base<T>::search;

    verify_forward_conv_int8(const tensor<T>& pinput,
                             const tensor<T>& pweights,
                             const miopen::ConvolutionDescriptor& pfilter,
                             int pbias   = 0,
                             int psearch = 0)
    {
        input   = pinput;
        weights = pweights;
        filter  = pfilter;
        bias    = pbias;
        search  = psearch;
    }

    tensor<float> cpu() const
    {
        auto rout = get_output_tensor_int8(filter, input, weights);

        if(filter.mode == miopenConvolution)
        {
            int in_h, in_w;
            std::tie(std::ignore, std::ignore, in_h, in_w) =
                miopen::tien<4>(input.desc.GetLengths());

            int wei_c, wei_h, wei_w;
            std::tie(std::ignore, wei_c, wei_h, wei_w) = miopen::tien<4>(weights.desc.GetLengths());

            rout.par_for_each([&](int n, int k, int ho, int wo) {
                const int start_h = ho * filter.GetConvStrides()[0] - filter.GetConvPads()[0];
                const int start_w = wo * filter.GetConvStrides()[1] - filter.GetConvPads()[1];

                double acc = bias;
                ford(wei_c, wei_h, wei_w)([&](int c, int x, int y) {
                    const int hi = start_h + x * filter.GetConvDilations()[0];
                    const int wi = start_w + y * filter.GetConvDilations()[1];
                    if(hi >= 0 && hi < in_h && wi >= 0 && wi < in_w)
                    {
                        acc += double(input(n, c, hi, wi)) * double(weights(k, c, x, y));
                    }
                });
                rout(n, k, ho, wo) = acc;
            });
        }

        return rout;
    }

    tensor<float> gpu() const
    {
        auto&& handle = get_handle();
        auto rout     = get_output_tensor_int8(filter, input, weights);

        auto in_dev  = handle.Write(input.data);
        auto wei_dev = handle.Write(weights.data);
        auto out_dev = handle.Write(rout.data);

        bool is_int8_pad4 = ((input.desc.GetLengths()[1]) % 4 != 0);

        std::vector<int> in_len(input.desc.GetLengths().begin(), input.desc.GetLengths().end()),
            wei_len(weights.desc.GetLengths().begin(), weights.desc.GetLengths().end());
        in_len[1]  = ((in_len[1] + 3) / 4) * 4;
        wei_len[1] = ((wei_len[1] + 3) / 4) * 4;

        miopen::TensorDescriptor input_int8pad4_desc;
        miopen::TensorDescriptor weight_int8pad4_desc;
        input_int8pad4_desc  = miopen::TensorDescriptor(miopenInt8, in_len.data(), in_len.size());
        weight_int8pad4_desc = miopen::TensorDescriptor(miopenInt8, wei_len.data(), wei_len.size());

        auto input_int8pad4   = tensor<T>{in_len};
        auto weights_int8pad4 = tensor<T>{wei_len};
        auto in_int8pad4_dev  = handle.Write(input_int8pad4.data);
        auto wei_int8pad4_dev = handle.Write(weights_int8pad4.data);

        if(is_int8_pad4)
        {
            float aph = 1.0;
            float bta = 0.0;
            miopen::TransformTensor(handle,
                                    &aph,
                                    input.desc,
                                    in_dev.get(),
                                    &bta,
                                    input_int8pad4_desc,
                                    in_int8pad4_dev.get());

            miopen::TransformTensor(handle,
                                    &aph,
                                    weights.desc,
                                    wei_dev.get(),
                                    &bta,
                                    weight_int8pad4_desc,
                                    wei_int8pad4_dev.get());
        }

        size_t workspace_size =
            filter.ForwardGetWorkSpaceSize(handle,
                                           (is_int8_pad4 ? weight_int8pad4_desc : weights.desc),
                                           (is_int8_pad4 ? input_int8pad4_desc : input.desc),
                                           rout.desc);

        std::vector<char> workspace(workspace_size);
        auto workspace_dev = workspace_size != 0 ? handle.Write(workspace) : nullptr;

        int ret_algo_count;
        miopenConvAlgoPerf_t perf;

        float alpha = 1, beta = 0;

        filter.FindConvFwdAlgorithm(handle,
                                    (is_int8_pad4 ? input_int8pad4_desc : input.desc),
                                    (is_int8_pad4 ? in_int8pad4_dev.get() : in_dev.get()),
                                    (is_int8_pad4 ? weight_int8pad4_desc : weights.desc),
                                    (is_int8_pad4 ? wei_int8pad4_dev.get() : wei_dev.get()),
                                    rout.desc,
                                    out_dev.get(),
                                    1,
                                    &ret_algo_count,
                                    &perf,
                                    workspace_dev.get(),
                                    workspace_size,
                                    search);

        filter.ConvolutionForward(handle,
                                  &alpha,
                                  (is_int8_pad4 ? input_int8pad4_desc : input.desc),
                                  (is_int8_pad4 ? in_int8pad4_dev.get() : in_dev.get()),
                                  (is_int8_pad4 ? weight_int8pad4_desc : weights.desc),
                                  (is_int8_pad4 ? wei_int8pad4_dev.get() : wei_dev.get()),
                                  perf.fwd_algo,
                                  &beta,
                                  rout.desc,
                                  out_dev.get(),
                                  workspace_dev.get(),
                                  workspace_size);

        rout.data = handle.Read<float>(out_dev, rout.data.size());

        return rout;
    }

    void fail(float = 0) const
    {
        std::cout << "Forward convolution: " << std::endl;
        this->conv_base<T>::fail();
    }
};

template <class T>
struct verify_backward_conv : conv_base<T>
{
    using conv_base<T>::input;
    using conv_base<T>::weights;
    using conv_base<T>::out;
    using conv_base<T>::filter;
    using conv_base<T>::bias;
    using conv_base<T>::search;

    verify_backward_conv(const tensor<T>& pinput,
                         const tensor<T>& pweights,
                         const tensor<T>& pout,
                         const miopen::ConvolutionDescriptor& pfilter,
                         int pbias   = 0,
                         int psearch = 0)
    {
        input   = pinput;
        weights = pweights;
        out     = pout;
        filter  = pfilter;
        bias    = pbias;
        search  = psearch;
    }

    tensor<T> cpu() const
    {
        auto rinput = input;
        std::fill(rinput.begin(), rinput.end(), 0);

        if(filter.mode == miopenTranspose)
        {
            int out_h, out_w;
            std::tie(std::ignore, std::ignore, out_h, out_w) =
                miopen::tien<4>(out.desc.GetLengths());

            int wei_k, wei_c, wei_h, wei_w;
            std::tie(wei_k, wei_c, wei_h, wei_w) = miopen::tien<4>(weights.desc.GetLengths());

            rinput.par_for_each([&](int n, int k, int j, int i) {
                const int start_y  = j * filter.GetConvStrides()[0] - filter.GetConvPads()[0];
                const int start_x  = i * filter.GetConvStrides()[1] - filter.GetConvPads()[1];
                const int group_id = k / (wei_k / filter.group_count);

                double acc = 0.0;
                ford(wei_c, wei_h, wei_w)([&](int c_grp, int y, int x) {
                    const int out_y  = start_y + y * filter.GetConvDilations()[0];
                    const int out_x  = start_x + x * filter.GetConvDilations()[1];
                    const int out_ch = group_id * wei_c + c_grp;
                    if(out_y >= 0 && out_y < out_h && out_x >= 0 && out_x < out_w)
                    {
                        acc +=
                            double(out(n, out_ch, out_y, out_x)) * double(weights(k, c_grp, y, x));
                    }
                });
                rinput(n, k, j, i) = acc;
            });
        }
        else
        {
            int in_n, in_c, in_h, in_w;
            std::tie(in_n, in_c, in_h, in_w) = miopen::tien<4>(rinput.desc.GetLengths());

            int wei_k, wei_c, wei_h, wei_w;
            std::tie(wei_k, wei_c, wei_h, wei_w) = miopen::tien<4>(weights.desc.GetLengths());

            int out_h, out_w;
            std::tie(std::ignore, std::ignore, out_h, out_w) =
                miopen::tien<4>(out.desc.GetLengths());

            par_ford(in_n, in_c, in_h, in_w)([&](int n, int c, int hi, int wi) {
                const int group_id = c / wei_c;
                double acc         = 0;
                ford(wei_k / filter.group_count, wei_h, wei_w)([&](int k, int y, int x) {
                    int h_ = filter.GetConvPads()[0] + hi - y * filter.GetConvDilations()[0];
                    int w_ = filter.GetConvPads()[1] + wi - x * filter.GetConvDilations()[1];

                    int ho = h_ / filter.GetConvStrides()[0];
                    int wo = w_ / filter.GetConvStrides()[1];

                    if(((ho * filter.GetConvStrides()[0] == h_) and
                        (wo * filter.GetConvStrides()[1] == w_)) and
                       ((ho >= 0 and ho < out_h) and (wo >= 0 and wo < out_w)))
                    {
                        const int out_ch = group_id * (wei_k / filter.group_count) + k;
                        const int wei_ch = c % wei_c;
                        acc +=
                            double(out(n, out_ch, ho, wo)) * double(weights(out_ch, wei_ch, y, x));
                    }
                });
                rinput(n, c, hi, wi) = acc;
            });
        }
        return rinput;
    }

    tensor<T> gpu() const
    {
        auto&& handle = get_handle();
        auto rinput   = input;
        std::fill(rinput.begin(), rinput.end(), 0);

        auto out_dev = handle.Write(out.data);
        auto wei_dev = handle.Write(weights.data);
        auto in_dev  = handle.Write(rinput.data);

        size_t workspace_size =
            filter.mode == miopenTranspose
                ? filter.ForwardGetWorkSpaceSize(handle, weights.desc, out.desc, rinput.desc)
                : filter.BackwardDataGetWorkSpaceSize(handle, weights.desc, out.desc, rinput.desc);

        std::vector<char> workspace(workspace_size);
        auto workspace_dev = workspace_size != 0 ? handle.Write(workspace) : nullptr;

        int ret_algo_count;
        miopenConvAlgoPerf_t perf;

        float alpha = 1, beta = 0;

        if(filter.mode == miopenTranspose)
        {
            filter.FindConvFwdAlgorithm(handle,
                                        out.desc,
                                        out_dev.get(),
                                        weights.desc,
                                        wei_dev.get(),
                                        rinput.desc,
                                        in_dev.get(),
                                        1,
                                        &ret_algo_count,
                                        &perf,
                                        workspace_dev.get(),
                                        workspace_size,
                                        search);

            filter.ConvolutionForward(handle,
                                      &alpha,
                                      out.desc,
                                      out_dev.get(),
                                      weights.desc,
                                      wei_dev.get(),
                                      perf.fwd_algo,
                                      &beta,
                                      rinput.desc,
                                      in_dev.get(),
                                      workspace_dev.get(),
                                      workspace_size);
        }
        else
        {
            filter.FindConvBwdDataAlgorithm(handle,
                                            out.desc,
                                            out_dev.get(),
                                            weights.desc,
                                            wei_dev.get(),
                                            rinput.desc,
                                            in_dev.get(),
                                            1,
                                            &ret_algo_count,
                                            &perf,
                                            workspace_dev.get(),
                                            workspace_size,
                                            search);

            filter.ConvolutionBackwardData(handle,
                                           &alpha,
                                           out.desc,
                                           out_dev.get(),
                                           weights.desc,
                                           wei_dev.get(),
                                           perf.bwd_data_algo,
                                           &beta,
                                           rinput.desc,
                                           in_dev.get(),
                                           workspace_dev.get(),
                                           workspace_size);
        }

        rinput.data = handle.Read<T>(in_dev, rinput.data.size());
        return rinput;
    }

    void fail(float) const
    {
        std::cout << "Backward convolution: " << std::endl;
        this->conv_base<T>::fail();
    }
};

template <class T>
struct verify_backward_weights_conv : conv_base<T>
{
    using conv_base<T>::input;
    using conv_base<T>::weights;
    using conv_base<T>::out;
    using conv_base<T>::filter;
    using conv_base<T>::bias;
    using conv_base<T>::search;

    verify_backward_weights_conv(const tensor<T>& pinput,
                                 const tensor<T>& pweights,
                                 const tensor<T>& pout,
                                 const miopen::ConvolutionDescriptor& pfilter,
                                 int pbias   = 0,
                                 int psearch = 0)
    {
        input   = pinput;
        weights = pweights;
        out     = pout;
        filter  = pfilter;
        bias    = pbias;
        search  = psearch;
    }

    tensor<T> cpu() const
    {
        auto rweights = weights;
        std::fill(rweights.begin(), rweights.end(), 0);

        int in_h, in_w;
        std::tie(std::ignore, std::ignore, in_h, in_w) = miopen::tien<4>(
            filter.mode == miopenTranspose ? out.desc.GetLengths() : input.desc.GetLengths());

        int wei_c, wei_h, wei_w;
        std::tie(std::ignore, wei_c, wei_h, wei_w) = miopen::tien<4>(rweights.desc.GetLengths());

        int out_n, out_c, out_h, out_w;
        std::tie(out_n, out_c, out_h, out_w) = miopen::tien<4>(
            filter.mode == miopenTranspose ? input.desc.GetLengths() : out.desc.GetLengths());

        int groups = filter.group_count;

        par_ford(out_c, wei_c, wei_h, wei_w)([&](int w, int k, int x, int y) {
            double acc         = 0.0;
            const int group_id = w / (out_c / groups);
            const int in_ch    = group_id * wei_c + k;
            ford(out_n, out_h, out_w)([&](int o, int i, int j) {
                const int start_x = i * filter.GetConvStrides()[0] - filter.GetConvPads()[0];
                const int start_y = j * filter.GetConvStrides()[1] - filter.GetConvPads()[1];
                const int in_x    = start_x + x * filter.GetConvDilations()[0];
                const int in_y    = start_y + y * filter.GetConvDilations()[1];
                if(in_x >= 0 && in_x < in_h && in_y >= 0 && in_y < in_w)
                {
                    acc += (filter.mode == miopenTranspose
                                ? double(out(o, in_ch, in_x, in_y)) * double(input(o, w, i, j))
                                : double(input(o, in_ch, in_x, in_y)) * double(out(o, w, i, j)));
                }
            });
            rweights(w, k, x, y) = acc;
        });
        return rweights;
    }

    tensor<T> gpu() const
    {
        auto&& handle = get_handle();
        auto rweights = weights;
        std::fill(rweights.begin(), rweights.end(), 0);

        auto out_dev = handle.Write(out.data);
        auto wei_dev = handle.Write(rweights.data);
        auto in_dev  = handle.Write(input.data);

        std::size_t workspace_size = filter.ConvolutionBackwardWeightsGetWorkSpaceSize(
            handle,
            filter.mode == miopenTranspose ? input.desc : out.desc,
            filter.mode == miopenTranspose ? out.desc : input.desc,
            rweights.desc);

        std::vector<char> workspace(workspace_size);
        auto workspace_dev = workspace_size != 0 ? handle.Write(workspace) : nullptr;

        int ret_algo_count;
        miopenConvAlgoPerf_t perf;

        float alpha = 1, beta = 0;
        filter.FindConvBwdWeightsAlgorithm(
            handle,
            filter.mode == miopenTranspose ? input.desc : out.desc,
            filter.mode == miopenTranspose ? in_dev.get() : out_dev.get(),
            filter.mode == miopenTranspose ? out.desc : input.desc,
            filter.mode == miopenTranspose ? out_dev.get() : in_dev.get(),
            rweights.desc,
            wei_dev.get(),
            1,
            &ret_algo_count,
            &perf,
            workspace_dev.get(),
            workspace_size,
            search);

        filter.ConvolutionBackwardWeights(
            handle,
            &alpha,
            filter.mode == miopenTranspose ? input.desc : out.desc,
            filter.mode == miopenTranspose ? in_dev.get() : out_dev.get(),
            filter.mode == miopenTranspose ? out.desc : input.desc,
            filter.mode == miopenTranspose ? out_dev.get() : in_dev.get(),
            perf.bwd_weights_algo,
            &beta,
            rweights.desc,
            wei_dev.get(),
            workspace_dev.get(),
            workspace_size);

        rweights.data = handle.Read<T>(wei_dev, rweights.data.size());
        return rweights;
    }

    void fail(float) const
    {
        std::cout << "Backward weights convolution: " << std::endl;
        this->conv_base<T>::fail();
    }
};

template <class T>
struct conv_driver : test_driver
{
    tensor<T> input;
    tensor<T> weights;
    miopen::ConvolutionDescriptor filter;
    std::string conv_mode;
    std::string pad_mode;
    std::vector<int> pads_strides_dilations;
    int groupCount{};
    bool do_forward          = true;
    bool do_backward_data    = true;
    bool do_backward_weights = true;
    int search               = 0;
    bool gen_float           = false;

    std::unordered_map<std::string, miopenConvolutionMode_t> cmode_lookup = {
        {"CONV", miopenConvolution},
        {"TRANS", miopenTranspose},
        {"CONVOLUTION", miopenConvolution},
        {"TRANSPOSE", miopenTranspose}};

    std::unordered_map<std::string, miopenPaddingMode_t> pmode_lookup = {
        {"SAME", miopenPaddingSame},
        {"VALID", miopenPaddingValid},
        {"DEFAULT", miopenPaddingDefault}};

    conv_driver()
    {
        add(input, "input", get_input_tensor());
        add(weights, "weights", get_weights_tensor());
        add(conv_mode, "cmode", generate_data({"conv"}));
        add(pad_mode, "pmode", generate_data({"default", "same", "valid"}));
        add(pads_strides_dilations,
            "pads_strides_dilations",
            generate_data(get_pads_strides_dilations()));
        add(groupCount, "group-count", generate_data({1}));
        add(do_forward, "disable-forward", set_value(false));
        add(do_backward_data, "disable-backward-data", set_value(false));
        add(do_backward_weights, "disable-backward-weights", set_value(false));
        add(search, "search", set_value(1));
        add(gen_float, "generate-float", set_value(true));
    }

    std::vector<std::vector<int>> get_pads_strides_dilations()
    {
        return {{0, 0, 1, 1, 1, 1},
                {0, 0, 2, 2, 1, 1},
                {1, 1, 1, 1, 1, 1},
                {1, 1, 2, 2, 1, 1},
                {2, 2, 1, 1, 1, 1},
                {3, 3, 2, 2, 1, 1},
                {0, 0, 1, 1, 2, 2},
                {1, 1, 2, 2, 3, 3},
                {3, 3, 2, 2, 4, 4},
                {0, 0, 1, 1, 1, 2},
                {1, 1, 2, 2, 2, 1},
                {2, 2, 1, 1, 4, 3},
                {3, 3, 2, 2, 3, 4}};
    }

    void run()
    {
        int input_c, input_h, input_w, wei_c, wei_k, wei_h, wei_w;
        std::tie(wei_k, wei_c, wei_h, wei_w) = miopen::tien<4>(weights.desc.GetLengths());
        std::tie(std::ignore, input_c, input_h, input_w) = miopen::tien<4>(input.desc.GetLengths());

        filter.mode         = cmode_lookup[miopen::ToUpper(conv_mode)];
        filter.paddingMode  = pmode_lookup[miopen::ToUpper(pad_mode)];
        filter.pads[0]      = pads_strides_dilations[0];
        filter.pads[1]      = pads_strides_dilations[1];
        filter.strides[0]   = pads_strides_dilations[2];
        filter.strides[1]   = pads_strides_dilations[3];
        filter.dilations[0] = pads_strides_dilations[4];
        filter.dilations[1] = pads_strides_dilations[5];
        filter.group_count  = std::max(static_cast<int>(groupCount), 1);

        // lack of transposeConv or groupConv for int8 type
        if(input.desc.GetType() == miopenInt8 &&
           (filter.mode == miopenTranspose || filter.group_count >= 2))
        {
            return;
        }

        // bwd53 kernel (large images supported) doesnt support stride !=1 and dialation and pad.
        if(input_w >= 2048 &&
           ((filter.GetConvStrides()[0] != 1) || (filter.GetConvStrides()[1] != 1) ||
            (filter.GetConvDilations()[0] != 1) || (filter.GetConvDilations()[1] != 1) ||
            (filter.GetConvPads()[1] != 0) || (filter.GetConvPads()[0] != 0)))
        {
            return;
        }

        if(((filter.mode == miopenTranspose) &&
            ((filter.group_count == 1 && input_c == wei_k) ||
             (filter.group_count >= 2 && wei_k % filter.group_count == 0))) ||
           ((filter.mode == miopenConvolution) &&
            ((filter.group_count == 1 && input_c == wei_c) ||
             (filter.group_count >= 2 && input_c % wei_c == 0))))
        {
            if(filter.mode == miopenConvolution &&
               ((filter.GetConvDilations()[0] == 1 && filter.GetConvDilations()[1] == 1) ||
                (wei_h == 1 && wei_w == 1)))
            {
                if(filter.paddingMode == miopenPaddingSame)
                {
                    if(filter.GetConvStrides()[0] == 0 || filter.GetConvStrides()[1] == 0)
                        return;
                    auto _pad_h =
                        (input_h % filter.GetConvStrides()[0] == 0)
                            ? (std::max(static_cast<int>(wei_h - filter.GetConvStrides()[0]), 0))
                            : (std::max(
                                  static_cast<int>(wei_h - (input_h % filter.GetConvStrides()[0])),
                                  0));
                    auto _pad_w =
                        (input_w % filter.GetConvStrides()[1] == 0)
                            ? (std::max(static_cast<int>(wei_w - filter.GetConvStrides()[1]), 0))
                            : (std::max(
                                  static_cast<int>(wei_w - (input_w % filter.GetConvStrides()[1])),
                                  0));

                    filter.pads[0] = _pad_h / 2;
                    filter.pads[1] = _pad_w / 2;

                    int out_h =
                        std::ceil(static_cast<double>(input_h) / filter.GetConvStrides()[0]);
                    int out_w =
                        std::ceil(static_cast<double>(input_w) / filter.GetConvStrides()[1]);

                    if(out_h <= 0 || out_w <= 0)
                        return;
                }
                else if(filter.paddingMode == miopenPaddingValid)
                {
                    if(filter.GetConvStrides()[0] == 0 || filter.GetConvStrides()[1] == 0)
                        return;
                    filter.pads[0] = 0;
                    filter.pads[1] = 0;

                    int out_h = std::ceil(static_cast<double>(input_h - wei_h + 1) /
                                          filter.GetConvStrides()[0]);
                    int out_w = std::ceil(static_cast<double>(input_w - wei_w + 1) /
                                          filter.GetConvStrides()[1]);

                    if(out_h <= 0 || out_w <= 0)
                        return;
                }
            }
            if(filter.mode == miopenTranspose)
            {
                filter.pads[0] = filter.GetConvStrides()[0] - 1;
                filter.pads[1] = filter.GetConvStrides()[1] - 1;
            }

            if(((filter.mode == miopenTranspose) &&
                ((filter.group_count == 1 &&
                  (input.desc.GetLengths().at(1) == weights.desc.GetLengths().at(0))) ||
                 (filter.group_count >= 2 &&
                  (weights.desc.GetLengths().at(0) % filter.group_count == 0)))) ||
               ((filter.mode == miopenConvolution) &&
                ((filter.group_count == 1 &&
                  (input.desc.GetLengths().at(1) == weights.desc.GetLengths().at(1))) ||
                 (filter.group_count >= 2 &&
                  (input.desc.GetLengths().at(1) % weights.desc.GetLengths().at(1) == 0)))))
            {
                auto output = get_output_tensor(filter, input, weights);

                auto gen_positive_value = [=](auto, auto, auto, auto) {
                    if(input.desc.GetType() == miopenInt8)
                        return scalar_gen_random_integer{0, 127}();
                    else
                        return gen_float ? scalar_gen_random_float{0, 1}()
                                         : scalar_gen_random_integer{
                                               1, miopen_type<T>{} == miopenHalf ? 4 : 16}();
                };

                auto gen_sign_value = [=](auto n, auto c, auto h, auto w) {
                    if(input.desc.GetType() == miopenInt8)
                        return (scalar_gen_random_integer{0, 127}() *
                                tensor_elem_gen_checkboard_sign{}(n, c, h, w));
                    else
                        return gen_float ? scalar_gen_random_float{-1, 1}()
                                         : scalar_gen_random_integer{1,
                                                                     miopen_type<T>{} == miopenHalf
                                                                         ? 4
                                                                         : 16}() *
                                               tensor_elem_gen_checkboard_sign{}(n, c, h, w);
                };

                bool skip_forward          = false;
                bool skip_backward_data    = false;
                bool skip_backward_weights = false;

#if TEST_DIRECT_SUPPORTED_CONFIG_ONLY
                if(input.desc.GetType() == miopenInt8)
                {
                    return;
                }
                if(input.desc.GetType() == miopenHalf && filter.mode == miopenConvolution)
                {
                    skip_forward = !is_direct_fwd_bwd_data_supported(
                        get_handle(), filter, input.desc, weights.desc, output.desc);

                    skip_backward_data = skip_forward;

                    skip_backward_weights = !is_direct_bwd_wrw_supported(
                        get_handle(), filter, input.desc, weights.desc, output.desc);
                }
#endif

                // bwd53 kernel (large images supported) doesnt support stride !=1 and dialation and
                // pad.
                if(input_w >= 2048 &&
                   ((filter.GetConvStrides()[0] != 1) || (filter.GetConvStrides()[1] != 1) ||
                    (filter.GetConvDilations()[0] != 1) || (filter.GetConvDilations()[1] != 1) ||
                    (filter.GetConvPads()[1] != 0) || (filter.GetConvPads()[0] != 0)))
                {
                    return;
                }

                // ToDo: workaround for workspace exceeding upperlimit issue
                if(input.desc.GetType() == miopenInt8 &&
                   !is_int8_workspace_valid(
                       get_handle(), filter, input.desc, weights.desc, output.desc))
                {
                    return;
                }

                input.generate(gen_positive_value);
                output.generate(gen_positive_value);
                weights.generate(gen_sign_value);

                if(do_forward && !skip_forward)
                {
                    if(input.desc.GetType() == miopenInt8)
                        verify(verify_forward_conv_int8<T>{input, weights, filter, 0, search});
                    else
                        verify(verify_forward_conv<T>{input, weights, filter, 0, search});
                }

                if(do_backward_data && !skip_backward_data && input.desc.GetType() != miopenInt8)
                {
                    verify(verify_backward_conv<T>{input, weights, output, filter, 0, search});
                }

                if(do_backward_weights && !skip_backward_weights &&
                   input.desc.GetType() != miopenInt8)
                {
                    output.generate(gen_sign_value);

                    verify(
                        verify_backward_weights_conv<T>{input, weights, output, filter, 0, search});
                }
            }
        }
    }
};

int main(int argc, const char* argv[]) { test_drive<conv_driver>(argc, argv); }
