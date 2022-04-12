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
#ifndef GUARD_CPU_BIAS_HPP
#define GUARD_CPU_BIAS_HPP

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

template <std::size_t NSpatialDim, typename Tout, typename Tbias>
void cpu_bias_forward_impl(tensor<Tout>& out, const tensor<Tbias>& bias)
{
    assert(out.desc.GetSize() == NSpatialDim + 2 and bias.desc.GetSize() == NSpatialDim + 2);
    assert(
        bias.desc.GetLengths()[0] == 1 && bias.desc.GetLengths()[1] == out.desc.GetLengths()[0] &&
        std::all_of(bias.desc.GetLengths().begin() + 2, bias.desc.GetLengths().end(), [](auto v) {
            return v == 1;
        }));

    out.par_for_each([&](auto out_n_id, auto out_k_id, auto... out_spatial_id_pack) {
        out(out_n_id, out_k_id, out_spatial_id_pack...) =
            double(out(out_n_id, out_k_id, out_spatial_id_pack...)) + double(bias.data[out_k_id]);
    });
}

template <std::size_t NSpatialDim, typename Tout, typename Tbias>
void cpu_bias_backward_data_impl(const tensor<Tout>& out, tensor<Tbias>& bias)
{
    assert(out.desc.GetSize() == NSpatialDim + 2 and bias.desc.GetSize() == NSpatialDim + 2);
    assert(
        bias.desc.GetLengths()[0] == 1 && bias.desc.GetLengths()[1] == out.desc.GetLengths()[0] &&
        std::all_of(bias.desc.GetLengths().begin() + 2, bias.desc.GetLengths().end(), [](auto v) {
            return v == 1;
        }));

    std::size_t out_n_len = out.desc.GetLengths()[0];
    std::size_t out_k_len = out.desc.GetLengths()[1];

    std::array<std::size_t, NSpatialDim> out_spatial_len{};
    std::copy_n(out.desc.GetLengths().begin() + 2, NSpatialDim, out_spatial_len.begin());

    par_ford(out_k_len)([&](auto out_k_id) {
        auto ford_out_n_spatial =
            miopen::unpacker(miopen::prepender(ford, out_n_len))(out_spatial_len);

        double acc = 0;
        ford_out_n_spatial([&](auto out_n_id, auto... out_spatial_id_pack) {
            acc += double(out(out_n_id, out_k_id, out_spatial_id_pack...));
        });

        bias.data[out_k_id] = acc;
    });
}

template <typename Tout, typename Tbias>
void cpu_bias_forward(tensor<Tout>& out, const tensor<Tbias>& bias)
{
    switch(out.desc.GetSize())
    {
    case 3: {
        cpu_bias_forward_impl<1>(out, bias);
        break;
    }
    case 4: {
        cpu_bias_forward_impl<2>(out, bias);
        break;
    }
    case 5: {
        cpu_bias_forward_impl<3>(out, bias);
        break;
    }
    case 6: {
        cpu_bias_forward_impl<4>(out, bias);
        break;
    }
    default: {
        MIOPEN_THROW("not belong to any case");
    }
    }
}

template <typename Tout, typename Tbias>
void cpu_bias_backward_data(const tensor<Tout>& out, tensor<Tbias>& bias)
{
    switch(out.desc.GetSize())
    {
    case 3: {
        cpu_bias_backward_data_impl<1>(out, bias);
        break;
    }
    case 4: {
        cpu_bias_backward_data_impl<2>(out, bias);
        break;
    }
    case 5: {
        cpu_bias_backward_data_impl<3>(out, bias);
        break;
    }
    case 6: {
        cpu_bias_backward_data_impl<4>(out, bias);
        break;
    }
    default: {
        MIOPEN_THROW("not belong to any case");
    }
    }
}
#endif
