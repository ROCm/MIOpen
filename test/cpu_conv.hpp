/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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
#ifndef GUARD_CPU_CONV_HPP
#define GUARD_CPU_CONV_HPP

#include "test.hpp"
#include <array>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <utility>

#include "tensor_holder.hpp"
#include <miopen/stringutils.hpp>
#include <miopen/functional.hpp>
#include <hip_float8.hpp>

template <class T, class... Ts>
static constexpr auto make_array(T x, Ts... xs)
{
    return std::array<T, 1 + sizeof...(Ts)>{{x, xs...}};
}

template <typename T>
struct PassThru
{
    T operator()(T t) { return t; }
};

template <typename Tin, typename Twei, typename Tout>
struct cpu_convolution_acc_type
{
    using type = double; // default using double as accumulator
};

template <>
struct cpu_convolution_acc_type<int8_t, int8_t, int32_t>
{
    using type = int32_t;
};

template <>
struct cpu_convolution_acc_type<int8_t, int8_t, float>
{
    using type = double;
};

template <std::size_t ConvDim,
          typename Tacc,
          typename FI,
          typename FW,
          typename Tin,
          typename Twei,
          typename Tout,
          typename Range>
void cpu_convolution_forward_impl(const tensor<Tin>& in,
                                  const tensor<Twei>& wei,
                                  tensor<Tout>& out,
                                  const Range& pads,
                                  const Range& strides,
                                  const Range& dilations,
                                  std::size_t group_count,
                                  FI fi = {},
                                  FW fw = {})
{
    static_assert(ConvDim > 0, "wrong! convolution dim should be larger than 0");
    assert(in.desc.GetNumDims() == ConvDim + 2 and wei.desc.GetNumDims() == ConvDim + 2 and
           out.desc.GetNumDims() == ConvDim + 2 and pads.size() == ConvDim and
           strides.size() == ConvDim and dilations.size() == ConvDim);
    std::size_t out_n_len = out.desc.GetLengths()[0];

    std::size_t wei_k_len = wei.desc.GetLengths()[0];
    std::size_t wei_c_len = wei.desc.GetLengths()[1];

    std::size_t vector_len = in.desc.GetVectorLength();

    std::array<std::size_t, ConvDim> in_spatial_len{};
    std::array<std::size_t, ConvDim> wei_spatial_len{};
    std::array<std::size_t, ConvDim> out_spatial_len{};

    std::copy_n(in.desc.GetLengths().begin() + 2, ConvDim, in_spatial_len.begin());
    std::copy_n(wei.desc.GetLengths().begin() + 2, ConvDim, wei_spatial_len.begin());
    std::copy_n(out.desc.GetLengths().begin() + 2, ConvDim, out_spatial_len.begin());

    if(wei.desc.GetLayout_str() == "CHWNc")
    {
        wei_c_len = wei.desc.GetLengths()[0];
        std::copy_n(wei.desc.GetLengths().begin() + 1, ConvDim, wei_spatial_len.begin());
        wei_k_len = wei.desc.GetLengths()[3];
    }

    std::size_t wei_k_len_per_group = wei_k_len / group_count;

    // f(x0, x1, xs...)
    // f1(xs...) = f(x0, x1, xs...)
    // f2(xs_array) = f1(xs...)
    auto par_ford_out_nk_spatial =
        miopen::unpacker(miopen::prepender(par_ford, out_n_len, wei_k_len))(out_spatial_len);

    par_ford_out_nk_spatial([&](std::size_t out_n_id,
                                std::size_t out_k_id,
                                auto... out_spatial_id_pack) {
        auto out_spatial_id = make_array(out_spatial_id_pack...);

        std::size_t group_id = out_k_id / wei_k_len_per_group;
        Tacc acc             = 0;

        ford(wei_c_len)([&](std::size_t wei_c_id) {
            std::size_t in_c_id = group_id * wei_c_len + wei_c_id;

            auto ford_wei_spatial = miopen::unpacker(ford)(wei_spatial_len);

            ford_wei_spatial([&](auto... wei_spatial_id_pack) {
                auto wei_spatial_id = make_array(wei_spatial_id_pack...);

                std::array<std::ptrdiff_t, ConvDim> in_spatial_id{};

                for(std::size_t i = 0; i < ConvDim; ++i)
                {
                    in_spatial_id[i] =
                        out_spatial_id[i] * strides[i] + wei_spatial_id[i] * dilations[i] - pads[i];
                }
                bool out_of_bound = false;
                for(std::size_t i = 0; i < ConvDim; ++i)
                {
                    out_of_bound = out_of_bound or
                                   (in_spatial_id[i] < 0 or in_spatial_id[i] >= in_spatial_len[i]);
                }
                if(!out_of_bound)
                {
                    if(vector_len > 1)
                    {
                        std::array<std::size_t, ConvDim + 3> in_id{};
                        in_id[1] = out_n_id;
                        in_id[2] = in_c_id;
                        std::copy_n(in_spatial_id.begin(), ConvDim, in_id.begin() + 3);
                        for(std::size_t i = 0; i < vector_len; i++)
                        {
                            in_id[0] = i;
                            acc += Tacc(in(in_id)) *
                                   Tacc(wei(i, out_k_id, wei_c_id, wei_spatial_id_pack...));
                        }
                    }
                    else
                    {
                        std::array<std::size_t, ConvDim + 2> in_id{};
                        in_id[0] = out_n_id;
                        in_id[1] = in_c_id;
                        std::copy_n(in_spatial_id.begin(), ConvDim, in_id.begin() + 2);
                        Tacc tmp1 = static_cast<Tacc>(fi(in(in_id)));
                        Tacc tmp2 =
                            static_cast<Tacc>(fw(wei(out_k_id, wei_c_id, wei_spatial_id_pack...)));
                        acc += tmp1 * tmp2;
                    }
                }
            });
        });
        if(vector_len > 1)
        {
            out(out_k_id % vector_len, out_n_id, out_k_id / vector_len, out_spatial_id_pack...) =
                static_cast<Tout>(acc);
        }
        else
        {
            out(out_n_id, out_k_id, out_spatial_id_pack...) = static_cast<Tout>(acc);
        }
    });
}

template <std::size_t ConvDim,
          typename Tacc,
          typename FW,
          typename FO,
          typename Tin,
          typename Twei,
          typename Tout,
          typename Range>
void cpu_convolution_backward_data_impl(tensor<Tin>& in,
                                        const tensor<Twei>& wei,
                                        const tensor<Tout>& out,
                                        const Range& pads,
                                        const Range& strides,
                                        const Range& dilations,
                                        std::size_t group_count,
                                        FW fw = {},
                                        FO fo = {})
{
    static_assert(ConvDim > 0, "wrong! convolution dim should be larger than 0");
    assert(in.desc.GetNumDims() == ConvDim + 2 and wei.desc.GetNumDims() == ConvDim + 2 and
           out.desc.GetNumDims() == ConvDim + 2 and pads.size() == ConvDim and
           strides.size() == ConvDim and dilations.size() == ConvDim);

    std::size_t in_n_len = in.desc.GetLengths()[0];
    std::size_t in_c_len = in.desc.GetLengths()[1];

    std::size_t wei_k_len = wei.desc.GetLengths()[0];
    std::size_t wei_c_len = wei.desc.GetLengths()[1];

    std::size_t wei_k_len_per_group = wei_k_len / group_count;

    std::array<std::size_t, ConvDim> in_spatial_len{};
    std::array<std::size_t, ConvDim> wei_spatial_len{};
    std::array<std::size_t, ConvDim> out_spatial_len{};

    std::copy_n(in.desc.GetLengths().begin() + 2, ConvDim, in_spatial_len.begin());
    std::copy_n(wei.desc.GetLengths().begin() + 2, ConvDim, wei_spatial_len.begin());
    std::copy_n(out.desc.GetLengths().begin() + 2, ConvDim, out_spatial_len.begin());

    auto par_ford_in_nc_spatial =
        miopen::unpacker(miopen::prepender(par_ford, in_n_len, in_c_len))(in_spatial_len);

    par_ford_in_nc_spatial(
        [&](std::size_t in_n_id, std::size_t in_c_id, auto... in_spatial_id_pack) {
            auto in_spatial_id = make_array(in_spatial_id_pack...);

            std::size_t group_id = in_c_id / wei_c_len;

            Tacc acc = 0;

            ford(wei_k_len_per_group)([&](std::size_t wei_k_id_inside_group) {
                auto ford_wei_spatial = miopen::unpacker(ford)(wei_spatial_len);

                ford_wei_spatial([&](auto... wei_spatial_id_pack) {
                    auto wei_spatial_id = make_array(wei_spatial_id_pack...);

                    std::array<ptrdiff_t, ConvDim> out_spatial_id_{};
                    std::array<ptrdiff_t, ConvDim> out_spatial_id{};

                    for(std::size_t i = 0; i < ConvDim; ++i)
                    {
                        out_spatial_id_[i] =
                            pads[i] + in_spatial_id[i] - wei_spatial_id[i] * dilations[i];
                        out_spatial_id[i] = out_spatial_id_[i] / strides[i];
                    }

                    bool use = true;
                    for(std::size_t i = 0; i < ConvDim; ++i)
                    {
                        use &= out_spatial_id_[i] % strides[i] == 0 and out_spatial_id[i] >= 0 and
                               out_spatial_id[i] < out_spatial_len[i];
                    }

                    if(use)
                    {
                        std::size_t out_k_id =
                            group_id * wei_k_len_per_group + wei_k_id_inside_group;
                        std::size_t wei_c_id = in_c_id % wei_c_len;

                        std::array<std::size_t, ConvDim + 2> out_id{};
                        out_id[0] = in_n_id;
                        out_id[1] = out_k_id;
                        std::copy_n(out_spatial_id.begin(), ConvDim, out_id.begin() + 2);
                        Tacc tmp1 = fo(out(out_id));
                        Tacc tmp2 = fw(wei(out_k_id, wei_c_id, wei_spatial_id_pack...));
                        acc += tmp1 * tmp2;
                    }
                });
            });
            // TODO: Why do we need a no-lint here ?
            in(in_n_id, in_c_id, in_spatial_id_pack...) = static_cast<Tout>(acc); // NOLINT
        });
}

template <std::size_t ConvDim,
          typename Tacc,
          typename FI,
          typename FO,
          typename Tin,
          typename Twei,
          typename Tout,
          typename Range>
void cpu_convolution_backward_weight_impl(const tensor<Tin>& in,
                                          tensor<Twei>& wei,
                                          const tensor<Tout>& out,
                                          const Range& pads,
                                          const Range& strides,
                                          const Range& dilations,
                                          std::size_t group_count,
                                          FI fi,
                                          FO fo)
{
    static_assert(ConvDim > 0, "wrong! convolution dim should be larger than 0");
    assert(in.desc.GetNumDims() == ConvDim + 2 and wei.desc.GetNumDims() == ConvDim + 2 and
           out.desc.GetNumDims() == ConvDim + 2 and pads.size() == ConvDim and
           strides.size() == ConvDim and dilations.size() == ConvDim);

    std::size_t out_n_len = out.desc.GetLengths()[0];

    std::size_t wei_k_len = wei.desc.GetLengths()[0];
    std::size_t wei_c_len = wei.desc.GetLengths()[1];

    std::size_t wei_k_len_per_group = wei_k_len / group_count;

    std::array<std::size_t, ConvDim> in_spatial_len{};
    std::array<std::size_t, ConvDim> wei_spatial_len{};
    std::array<std::size_t, ConvDim> out_spatial_len{};

    std::copy_n(in.desc.GetLengths().begin() + 2, ConvDim, in_spatial_len.begin());
    std::copy_n(wei.desc.GetLengths().begin() + 2, ConvDim, wei_spatial_len.begin());
    std::copy_n(out.desc.GetLengths().begin() + 2, ConvDim, out_spatial_len.begin());

    auto par_ford_wei_kc_spatial =
        miopen::unpacker(miopen::prepender(par_ford, wei_k_len, wei_c_len))(wei_spatial_len);

    par_ford_wei_kc_spatial(
        [&](std::size_t wei_k_id, std::size_t wei_c_id, auto... wei_spatial_id_pack) {
            auto wei_spatial_id = make_array(wei_spatial_id_pack...);

            std::size_t group_id = wei_k_id / wei_k_len_per_group;
            std::size_t in_c_id  = group_id * wei_c_len + wei_c_id;

            Tacc acc = 0;

            ford(out_n_len)([&](std::size_t out_n_id) {
                auto ford_out_spatial = miopen::unpacker(ford)(out_spatial_len);

                ford_out_spatial([&](auto... out_spatial_id_pack) {
                    auto out_spatial_id = make_array(out_spatial_id_pack...);

                    std::array<std::ptrdiff_t, ConvDim> in_spatial_id{};

                    for(std::size_t i = 0; i < ConvDim; ++i)
                    {
                        in_spatial_id[i] = out_spatial_id[i] * strides[i] +
                                           wei_spatial_id[i] * dilations[i] - pads[i];
                    }

                    bool out_of_bound = false;
                    for(std::size_t i = 0; i < ConvDim; ++i)
                    {
                        out_of_bound = out_of_bound or (in_spatial_id[i] < 0 or
                                                        in_spatial_id[i] >= in_spatial_len[i]);
                    }

                    if(!out_of_bound)
                    {
                        std::array<std::size_t, ConvDim + 2> in_id{};
                        in_id[0] = out_n_id;
                        in_id[1] = in_c_id;
                        std::copy_n(in_spatial_id.begin(), ConvDim, in_id.begin() + 2);
                        Tacc tmp1 = fi(in(in_id));
                        Tacc tmp2 = fo(out(out_n_id, wei_k_id, out_spatial_id_pack...));
                        acc += tmp1 * tmp2;
                    }
                });

                wei(wei_k_id, wei_c_id, wei_spatial_id_pack...) = static_cast<Twei>(acc);
            });
        });
}

template <typename Tin,
          typename Twei,
          typename Tout,
          typename Range,
          typename Tacc = double,
          typename FI   = PassThru<Tin>,
          typename FW   = PassThru<Twei>>
void cpu_convolution_forward(std::size_t spatial_dim,
                             const tensor<Tin>& in,
                             const tensor<Twei>& wei,
                             tensor<Tout>& out,
                             const Range& pads,
                             const Range& strides,
                             const Range& dilations,
                             std::size_t group_count,
                             FI fi = {},
                             FW fw = {})
{
    switch(spatial_dim)
    {
    case 1: {
        cpu_convolution_forward_impl<1, Tacc>(
            in, wei, out, pads, strides, dilations, group_count, fi, fw);
        break;
    }
    case 2: {
        cpu_convolution_forward_impl<2, Tacc>(
            in, wei, out, pads, strides, dilations, group_count, fi, fw);
        break;
    }
    case 3: {
        cpu_convolution_forward_impl<3, Tacc>(
            in, wei, out, pads, strides, dilations, group_count, fi, fw);
        break;
    }
    case 4: {
        cpu_convolution_forward_impl<4, Tacc>(
            in, wei, out, pads, strides, dilations, group_count, fi, fw);
        break;
    }
    default: {
        MIOPEN_THROW("not belong to any case");
    }
    }
}

template <typename Tin,
          typename Twei,
          typename Tout,
          typename Range,
          typename Tacc = double,
          typename FW   = PassThru<Twei>,
          typename FO   = PassThru<Tout>>
void cpu_convolution_backward_data(std::size_t spatial_dim,
                                   tensor<Tin>& in,
                                   const tensor<Twei>& wei,
                                   const tensor<Tout>& out,
                                   const Range& pads,
                                   const Range& strides,
                                   const Range& dilations,
                                   std::size_t group_count,
                                   FW fw = {},
                                   FO fo = {})
{
    switch(spatial_dim)
    {
    case 1: {
        cpu_convolution_backward_data_impl<1, Tacc>(
            in, wei, out, pads, strides, dilations, group_count, fw, fo);
        break;
    }
    case 2: {
        cpu_convolution_backward_data_impl<2, Tacc>(
            in, wei, out, pads, strides, dilations, group_count, fw, fo);
        break;
    }
    case 3: {
        cpu_convolution_backward_data_impl<3, Tacc>(
            in, wei, out, pads, strides, dilations, group_count, fw, fo);
        break;
    }
    case 4: {
        cpu_convolution_backward_data_impl<4, Tacc>(
            in, wei, out, pads, strides, dilations, group_count, fw, fo);
        break;
    }
    default: {
        MIOPEN_THROW("not belong to any case");
    }
    }
}

template <typename Tin,
          typename Twei,
          typename Tout,
          typename Range,
          typename Tacc = double,
          typename FI   = PassThru<Tin>,
          typename FO   = PassThru<Tout>>
void cpu_convolution_backward_weight(std::size_t spatial_dim,
                                     const tensor<Tin>& in,
                                     tensor<Twei>& wei,
                                     const tensor<Tout>& out,
                                     const Range& pads,
                                     const Range& strides,
                                     const Range& dilations,
                                     std::size_t group_count,
                                     FI fi = {},
                                     FO fo = {})
{
    switch(spatial_dim)
    {
    case 1: {
        cpu_convolution_backward_weight_impl<1, Tacc>(
            in, wei, out, pads, strides, dilations, group_count, fi, fo);
        break;
    }
    case 2: {
        cpu_convolution_backward_weight_impl<2, Tacc>(
            in, wei, out, pads, strides, dilations, group_count, fi, fo);
        break;
    }
    case 3: {
        cpu_convolution_backward_weight_impl<3, Tacc>(
            in, wei, out, pads, strides, dilations, group_count, fi, fo);
        break;
    }
    case 4: {
        cpu_convolution_backward_weight_impl<4, Tacc>(
            in, wei, out, pads, strides, dilations, group_count, fi, fo);
        break;
    }
    default: {
        MIOPEN_THROW("not belong to any case");
    }
    }
}
#endif
