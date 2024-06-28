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

#include "tensor_holder.hpp"
#include "tensor_view.hpp"
#include <miopen/cumulative_reduction/utils.hpp>

#include <limits>

inline constexpr void update() {}

template <typename T, typename... Ts>
inline constexpr void update(T& a, T b, Ts&... c, Ts... d)
{
    a = b;
    update(c..., d...);
}

inline constexpr bool greater() { return true; }

template <typename T, typename... Ts>
inline constexpr bool greater(T& a, T b, Ts&... c, Ts... d)
{
    if(a != b)
        return a > b;
    return greater(c..., d...);
}

template <typename T, typename... Ts>
struct reduce_func_base
{
    reduce_func_base(){};
    virtual ~reduce_func_base(){};
    virtual inline bool isbetter(const T& /*a*/, const T& /*b*/) const { return true; }
    virtual inline void combine(T& a, T b) const { a = b; }
    inline constexpr void calculate(T& a, T b, Ts&... c, Ts... d) const
    {
        if(isbetter(b, a) || (isbetter(a, b) == isbetter(b, a) && greater(c..., d...)))
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

template <miopenCumOp_t OP, class T>
void cpu_cumulative_reduction_forward(const tensor<T> input,
                                      tensor<T>& ref_output,
                                      tensor<int>& ref_indices,
                                      const int dim,
                                      const bool exclusive,
                                      const bool reverse,
                                      const bool has_output  = true,
                                      const bool has_indices = true)
{
    const auto ndims    = input.desc.GetSize();
    const auto true_dim = ((dim % ndims) + ndims) % ndims;

    std::cout << "True dim:" << true_dim << std::endl;

    auto input_tv = miopen::solver::cumulative_reduction::get_inner_expanded_tv<5>(input.desc);
    auto output_tv =
        miopen::solver::cumulative_reduction::get_inner_expanded_tv<5>(ref_output.desc);
    auto indices_tv =
        miopen::solver::cumulative_reduction::get_inner_expanded_tv<5>(ref_indices.desc);

    auto size       = input.desc.GetElementSize();
    auto inner_size = input.desc.GetLengths()[true_dim];
    auto outer_size = size / inner_size;

    auto op_worker = reduce_func<OP, float, int>{};

    tensor_view_t<5> ignore_dim_input_tv = input_tv;
    ignore_dim_input_tv.size[dim]        = 1;

    par_ford(outer_size)([&](int gid) {
        auto tensor_layout = tensor_layout_t<5>(ignore_dim_input_tv, gid);
        float tmp_val      = op_worker.START_VAL;
        int tmp_idx;

        ford(inner_size)([&](int idx) {
            tensor_layout.layout[true_dim] = (reverse ? input_tv.size[true_dim] - idx - 1 : idx);
            float val = static_cast<float>(input[input_tv.get_tensor_view_idx(tensor_layout)]);

            op_worker.calculate(tmp_val, val, tmp_idx, idx);

            if(has_output)
                ref_output[output_tv.get_tensor_view_idx(tensor_layout)] = static_cast<T>(tmp_val);
            if(has_indices)
                ref_indices[indices_tv.get_tensor_view_idx(tensor_layout)] = tmp_idx;
        });
    });

    std::cout << "CPU Finish" << std::endl;
}
