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
#include <miopen/functional.hpp>

template <class F>
void visit_tensor_size(std::size_t n, F f)
{
    switch(n)
    {
    case 0:
    {
        f(std::integral_constant<std::size_t, 0>{});
        break;
    }
    case 1:
    {
        f(std::integral_constant<std::size_t, 1>{});
        break;
    }
    case 2:
    {
        f(std::integral_constant<std::size_t, 2>{});
        break;
    }
    case 3:
    {
        f(std::integral_constant<std::size_t, 3>{});
        break;
    }
    case 4:
    {
        f(std::integral_constant<std::size_t, 4>{});
        break;
    }
    case 5:
    {
        f(std::integral_constant<std::size_t, 5>{});
        break;
    }
    default: throw std::runtime_error("Unknown tensor size");
    }
}

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

    tensor(std::size_t n, std::size_t c, std::size_t h, std::size_t w)
        : desc(miopenFloat, {n, c, h, w}), data(n * c * h * w)
    {
    }

    tensor(std::size_t n) : desc(miopenFloat, {n}), data(n) {}

    tensor(miopen::TensorDescriptor rhs) : desc(std::move(rhs))
    {
        data.resize(desc.GetElementSpace());
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
        auto assign   = [&](T x) {
            assert(iterator < data.end());
            *iterator = x;
            ++iterator;
        };
        this->for_each(miopen::compose(assign, std::move(g)));
    }

    template <class Loop, class F>
    struct for_each_unpacked
    {
        Loop loop;
        F f;
        template <class... Ts>
        auto operator()(Ts... xs) const -> decltype(f(xs...), void())
        {
            loop(xs...)(std::move(f));
        }

        struct any
        {
            any() {}
            template <class X>
            any(X)
            {
            }
        };

        void operator()(any = {},
                        any = {},
                        any = {},
                        any = {},
                        any = {},
                        any = {},
                        any = {},
                        any = {},
                        any = {}) const
        {
            throw std::runtime_error("Arguments to for_each do not match tensor size");
        }
    };

    struct for_each_handler
    {
        template <class Self, class Loop, class F, class Size>
        void operator()(Self* self, Loop loop, F f, Size size) const
        {
            auto dims = miopen::tien<size>(self->desc.GetLengths());
            miopen::unpack(for_each_unpacked<Loop, F>{loop, std::move(f)}, dims);
        }
    };

    template <class F>
    void for_each(F f) const
    {
        visit_tensor_size(
            desc.GetLengths().size(),
            std::bind(for_each_handler{}, this, ford, std::move(f), std::placeholders::_1));
    }

    template <class F>
    void par_for_each(F f) const
    {
        visit_tensor_size(
            desc.GetLengths().size(),
            std::bind(for_each_handler{}, this, par_ford, std::move(f), std::placeholders::_1));
    }

    template <class... Ts>
    T& operator()(Ts... xs)
    {
        assert(this->desc.GetIndex(xs...) < data.size());
        return this->data[this->desc.GetIndex(xs...)];
    }

    template <class... Ts>
    const T& operator()(Ts... xs) const
    {
        assert(this->desc.GetIndex(xs...) < data.size());
        return this->data[this->desc.GetIndex(xs...)];
    }

    T& operator[](std::size_t i) { return data.at(i); }

    const T& operator[](std::size_t i) const { return data.at(i); }

    typename std::vector<T>::iterator begin() { return data.begin(); }

    typename std::vector<T>::iterator end() { return data.end(); }

    typename std::vector<T>::const_iterator begin() const { return data.begin(); }

    typename std::vector<T>::const_iterator end() const { return data.end(); }
};

template <class T, class G>
tensor<T> make_tensor(std::initializer_list<std::size_t> dims, G g)
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
