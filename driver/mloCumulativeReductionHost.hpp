/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#pragma once

#include <../test/ford.hpp>

#include <miopen/tensor.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <miopen/cumulative_reduction/utils.hpp>

#include <limits>

inline constexpr void update() {}

template <typename T, typename... Ts>
inline constexpr void update(T& a, T b, Ts&... c, Ts... d)
{
    a = b;
    update(c..., d...);
}

template <typename T, typename... Ts>
struct reduce_func_base
{
    reduce_func_base(){};
    virtual ~reduce_func_base(){};
    virtual inline bool isbetter(const T& /*a*/, const T& /*b*/) const { return false; }
    virtual inline void combine(T& a, T b) const { a = b; }
    inline constexpr void calculate(T& a, T b, Ts&... c, Ts... d) const
    {
        if(!isbetter(a, b))
        {
            combine(a, b);
            update(c..., d...);
        }
    }
};

template <miopenCumOp_t OP, typename T, typename... Ts>
struct reduce_func : reduce_func_base<T, Ts...>
{
    virtual ~reduce_func(){};
};

template <typename T, typename... Ts>
struct reduce_func<MIOPEN_CUM_MAX, T, Ts...> : reduce_func_base<T, Ts...>
{
    const float START_VAL = -std::numeric_limits<float>::max();
    inline bool isbetter(const T& a, const T& b) const { return a > b; }
};

template <typename T, typename... Ts>
struct reduce_func<MIOPEN_CUM_MIN, T, Ts...> : reduce_func_base<T, Ts...>
{
    const float START_VAL = std::numeric_limits<float>::max();
    inline bool isbetter(const T& a, const T& b) const { return a < b; }
};

template <typename T, typename... Ts>
struct reduce_func<MIOPEN_CUM_SUM, T, Ts...> : reduce_func_base<T, Ts...>
{
    const float START_VAL = 0.0f;
    inline void combine(T& a, T b) const { a += b; }
};

template <typename T, typename... Ts>
struct reduce_func<MIOPEN_CUM_PROD, T, Ts...> : reduce_func_base<T, Ts...>
{
    const float START_VAL = 1.0f;
    inline void combine(T& a, T b) const { a *= b; }
};

template <miopenCumOp_t OP, typename Tgpu, typename Tcheck>
int32_t mloCumulativeReductionForwardRunHost(const miopenTensorDescriptor_t inputDesc,
                                             const miopenTensorDescriptor_t outputDesc,
                                             const miopenTensorDescriptor_t indicesDesc,
                                             const Tgpu* input,
                                             Tcheck* output_host,
                                             int* indices_host,
                                             const int dim,
                                             const bool exclusive,
                                             const bool reverse)
{
    const int ndims     = miopen::deref(inputDesc).GetNumDims();
    const auto exec_dim = ((dim % ndims) + ndims) % ndims;

    auto input_tv   = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto output_tv  = miopen::get_inner_expanded_tv<5>(miopen::deref(outputDesc));
    auto indices_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(indicesDesc));

    auto size       = miopen::deref(inputDesc).GetElementSize();
    auto inner_size = miopen::deref(inputDesc).GetLengths()[exec_dim];
    auto outer_size = size / inner_size;

    auto op_worker = reduce_func<OP, float, int>{};

    tensor_view_t<5> ignore_dim_input_tv = input_tv;
    ignore_dim_input_tv.size[exec_dim]   = 1;

    par_ford(outer_size)([&](int gid) {
        auto tensor_layout = tensor_layout_t<5>(ignore_dim_input_tv, gid);
        float cum_val      = op_worker.START_VAL;
        int cum_idx        = (reverse ? input_tv.size[exec_dim] - 1 : 0);

        ford(inner_size)([&](int idx) {
            int tmp_idx =
                (reverse ? input_tv.size[exec_dim] - (idx - exclusive) - 1 : (idx - exclusive));
            float tmp_val = op_worker.START_VAL;
            if(0 <= tmp_idx && tmp_idx < inner_size)
            {
                tensor_layout.layout[exec_dim] = tmp_idx;
                tmp_val = static_cast<float>(input[input_tv.get_tensor_view_idx(tensor_layout)]);
            }

            op_worker.calculate(cum_val, tmp_val, cum_idx, tmp_idx);

            tensor_layout.layout[exec_dim] = (reverse ? input_tv.size[exec_dim] - idx - 1 : idx);
            if(output_host)
                output_host[output_tv.get_tensor_view_idx(tensor_layout)] =
                    static_cast<Tcheck>(cum_val);
            if(indices_host)
                indices_host[indices_tv.get_tensor_view_idx(tensor_layout)] = cum_idx;
        });
    });

    return miopenStatusSuccess;
}
