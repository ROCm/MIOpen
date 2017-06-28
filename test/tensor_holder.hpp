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
#ifndef GUARD_TENSOR_HOLDER_HPP
#define GUARD_TENSOR_HOLDER_HPP

#include "ford.hpp"
#include "network_data.hpp"
#include <miopen/tensor.hpp>

template <class T>
struct tensor
{
    miopen::TensorDescriptor desc;
    std::vector<T> data;

    tensor() {}

    template <class X>
    tensor(const std::vector<X>& dims)
        : desc(miopenFloat, dims.data(), static_cast<int>(dims.size())), data(desc.GetElementSize())
    {
    }

    tensor(int n, int c, int h, int w) : desc(miopenFloat, {n, c, h, w}), data(n * c * h * w) {}

    tensor(miopen::TensorDescriptor rhs) : desc(std::move(rhs))
    {
        data.resize(desc.GetElementSize());
    }

    template <class G>
    tensor& generate(G g) &
    {
        this->generate_impl(g);
        return *this;
    }

    template <class G>
    tensor&& generate(G g) &&
    {
        this->generate_impl(g);
        return std::move(*this);
    }

    template <class G>
    void generate_impl(G g)
    {
        auto iterator = data.begin();
        this->for_each([&](int i, int j, int k, int m) {
            assert(iterator < data.end());
            *iterator = g(i, j, k, m);
            ++iterator;
        });
    }

    template <class F>
    void for_each(F f) const
    {
        int n, c, h, w;
        std::tie(n, c, h, w) = miopen::tie4(desc.GetLengths());
        ford(n, c, h, w)(std::move(f));
    }

    template <class F>
    void par_for_each(F f) const
    {
        int n, c, h, w;
        std::tie(n, c, h, w) = miopen::tie4(desc.GetLengths());
        par_ford(n, c, h, w)(std::move(f));
    }

    T& operator()(int n, int c, int h, int w)
    {
        assert(this->desc.GetIndex(n, c, h, w) < data.size());
        return this->data[this->desc.GetIndex(n, c, h, w)];
    }

    const T& operator()(int n, int c, int h, int w) const
    {
        assert(this->desc.GetIndex(n, c, h, w) < data.size());
        return this->data[this->desc.GetIndex(n, c, h, w)];
    }

    T& operator[](std::size_t i) { return data.at(i); }

    const T& operator[](std::size_t i) const { return data.at(i); }

    typename std::vector<T>::iterator begin() { return data.begin(); }

    typename std::vector<T>::iterator end() { return data.end(); }

    typename std::vector<T>::const_iterator begin() const { return data.begin(); }

    typename std::vector<T>::const_iterator end() const { return data.end(); }
};

template <class T, class G>
tensor<T> make_tensor(std::initializer_list<int> dims, G g)
{
    // TODO: Compute float
    return tensor<T>{miopen::TensorDescriptor{miopenFloat, dims}}.generate(g);
}

template <class T, class X>
tensor<T> make_tensor(const std::vector<X>& dims)
{
    // TODO: Compute float
    return tensor<T>{
        miopen::TensorDescriptor{miopenFloat, dims.data(), static_cast<int>(dims.size())}};
}

template <class T, class X, class G>
tensor<T> make_tensor(const std::vector<X>& dims, G g)
{
    return make_tensor<T>(dims).generate(g);
}

struct tensor_generate
{
    template <class Tensor, class G>
    Tensor&& operator()(Tensor&& t, G g) const
    {
        return std::forward<Tensor>(t.generate(g));
    }
};

template <class F>
struct protect_void_fn
{
    F f;
    protect_void_fn(F x) : f(std::move(x)) {}

    // template<class... Ts>
    // auto operator()(Ts&&... xs) const MIOPEN_RETURNS
    // (f(std::forward<Ts>(xs)...));

    template <class... Ts>
    void operator()(Ts&&... xs) const
    {
        f(std::forward<Ts>(xs)...);
    }
};

template <class F>
protect_void_fn<F> protect_void(F f)
{
    return {std::move(f)};
}

struct cross_args_apply
{
    template <class F, class T, class... Ts>
    void operator()(F f, T&& x, Ts&&... xs) const
    {
        miopen::each_args(std::bind(f, std::forward<T>(x), std::placeholders::_1),
                          std::forward<Ts>(xs)...);
    }
};

template <class F, class... Ts>
void cross_args(F f, Ts&&... xs)
{
    miopen::each_args(std::bind(cross_args_apply{},
                                protect_void(std::move(f)),
                                std::placeholders::_1,
                                std::forward<Ts>(xs)...),
                      std::forward<Ts>(xs)...);
}

template <class T>
struct generate_both_visitation
{
    template <class F, class G1, class G2>
    void operator()(F f, G1 g1, G2 g2) const
    {
        for(auto&& input : get_inputs())
            for(auto&& weights : get_weights())
                if(input.at(1) == weights.at(1)) // channels must match
                    f(make_tensor<T>(input, g1), make_tensor<T>(weights, g2));
    }
};

template <class T, class F, class... Gs>
void generate_binary_all(F f, Gs... gs)
{
    cross_args(std::bind(generate_both_visitation<T>{},
                         protect_void(f),
                         std::placeholders::_1,
                         std::placeholders::_2),
               gs...);
}

template <class T, class F, class G>
void generate_binary_one(F f, std::vector<int> input, std::vector<int> weights, G g)
{
    f(make_tensor<T>(input, g), make_tensor<T>(weights, g));
}

template <class T>
struct generate_activ_visitation
{
    template <class F, class G>
    void operator()(F f, G g) const
    {
        for(auto&& input : get_inputs())
            f(make_tensor<T>(input, g));
    }
};

template <class T, class F, class... Gs>
void generate_unary_all(F f, Gs... gs)
{
    miopen::each_args(
        std::bind(generate_activ_visitation<T>{}, protect_void(f), std::placeholders::_1), gs...);
}

template <class T, class F, class G>
void generate_unary_one(F f, std::vector<int> input, G g)
{
    f(make_tensor<T>(input, g));
}

#endif
