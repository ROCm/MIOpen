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
#include <miopen/type_name.hpp>
#include <miopen/each_args.hpp>
#include <miopen/bfloat16.hpp>

#include <half.hpp>
#include <iomanip>
#include <fstream>

template <class F>
void visit_tensor_size(std::size_t n, F f)
{
    switch(n)
    {
    case 0: {
        f(std::integral_constant<std::size_t, 0>{});
        break;
    }
    case 1: {
        f(std::integral_constant<std::size_t, 1>{});
        break;
    }
    case 2: {
        f(std::integral_constant<std::size_t, 2>{});
        break;
    }
    case 3: {
        f(std::integral_constant<std::size_t, 3>{});
        break;
    }
    case 4: {
        f(std::integral_constant<std::size_t, 4>{});
        break;
    }
    case 5: {
        f(std::integral_constant<std::size_t, 5>{});
        break;
    }
    default: throw std::runtime_error("Unknown tensor size");
    }
}

template <class T>
struct miopen_type;

template <>
struct miopen_type<float> : std::integral_constant<miopenDataType_t, miopenFloat>
{
};

template <>
struct miopen_type<double> : std::integral_constant<miopenDataType_t, miopenDouble>
{
};

template <>
struct miopen_type<half_float::half> : std::integral_constant<miopenDataType_t, miopenHalf>
{
};
template <>
struct miopen_type<bfloat16> : std::integral_constant<miopenDataType_t, miopenBFloat16>
{
};

template <>
struct miopen_type<int8_t> : std::integral_constant<miopenDataType_t, miopenInt8>
{
};

template <>
struct miopen_type<int> : std::integral_constant<miopenDataType_t, miopenInt32>
{
};

template <class T>
struct tensor
{
    miopen::TensorDescriptor desc;
    std::vector<T> data;

    tensor() : desc(miopen_type<T>{}, {}) {}

    template <class X>
    tensor(const std::vector<X>& dims) : desc(miopen_type<T>{}, dims), data(desc.GetElementSpace())
    {
    }

    template <class X>
    tensor(const std::vector<X>& dims, const std::vector<X>& strides)
        : desc(miopen_type<T>{}, dims, strides), data(desc.GetElementSpace())
    {
        assert(dims.size() == strides.size());
    }

    template <class X>
    tensor(miopenDataType_t t, miopenTensorLayout_t layout, const std::vector<X>& dims)
        : desc(t, layout, dims), data(desc.GetElementSpace())
    {
    }

    template <class X>
    tensor(miopenDataType_t t,
           miopenTensorLayout_t layout,
           const std::vector<X>& dims,
           const std::vector<X>& strides)
        : desc(t, layout, dims, strides), data(desc.GetElementSpace())
    {
        assert(dims.size() == strides.size());
    }

    tensor(std::size_t n, std::size_t c, std::size_t h, std::size_t w)
        : desc(miopen_type<T>{}, {n, c, h, w}), data(n * c * h * w)
    {
    }

    tensor(miopenDataType_t t,
           miopenTensorLayout_t layout,
           std::size_t n,
           std::size_t c,
           std::size_t h,
           std::size_t w)
        : desc(t, layout, {n, c, h, w}), data(desc.GetElementSpace())
    {
    }

    tensor(std::size_t n, std::size_t c, std::size_t d, std::size_t h, std::size_t w)
        : desc(miopen_type<T>{}, {n, c, d, h, w}), data(n * c * d * h * w)
    {
    }

    tensor(std::size_t n) : desc(miopen_type<T>{}, {n}), data(n) {}

    tensor(miopen::TensorDescriptor rhs) : desc(std::move(rhs))
    {
        assert(rhs.GetType() == miopen_type<T>{} ||
               ((miopen_type<T>{} == miopenInt8 || miopen_type<T>{} == miopenInt8x4) &&
                rhs.GetType() == miopenFloat));
        data.resize(desc.GetElementSpace());
    }

    template <class G>
    tensor& generate(G g) &
    {
        if(this->desc.GetVectorLength() > 1)
            this->generate_vect_impl(g);
        else
            this->generate_impl(g);
        return *this;
    }

    template <class G>
    tensor&& generate(G g) &&
    {
        if(this->desc.GetVectorLength() > 1)
            this->generate_vect_impl(g);
        else
            this->generate_impl(g);
        return std::move(*this);
    }

    template <class G>
    void generate_impl(G g)
    {
        auto seed = std::accumulate(desc.GetLengths().begin(),
                                    desc.GetLengths().end(),
                                    std::size_t{521288629},
                                    [](auto x, auto y) {
                                        x ^= x << 1U;
                                        return x ^ y;
                                    });
        seed ^= data.size();
        seed ^= desc.GetLengths().size();
        std::srand(seed);
        auto iterator = data.begin();
        auto assign   = [&](T x) {
            *iterator = x;
            ++iterator;
        };
        this->for_each(
            miopen::compose(miopen::compose(assign, miopen::cast_to<T>()), std::move(g)));
    }

    template <class G>
    void generate_vect_impl(G g)
    {
        auto seed = std::accumulate(desc.GetLengths().begin(),
                                    desc.GetLengths().end(),
                                    std::size_t{521288629},
                                    [](auto x, auto y) {
                                        x ^= x << 1U;
                                        return x ^ y;
                                    });
        seed ^= data.size();
        seed ^= desc.GetLengths().size();
        std::srand(seed);
        auto iterator     = data.begin();
        auto vectorLength = desc.GetVectorLength();
        auto assign       = [&](T x) {
            assert(iterator < data.end());
            // for debugging
            for(auto i = 0; i < vectorLength; i++)
            {
                *(iterator + i) = x;
            }
            iterator += vectorLength;
        };
        this->for_each(
            miopen::compose(miopen::compose(assign, miopen::cast_to<T>()), std::move(g)));
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
            throw std::runtime_error(
                "Arguments to for_each do not match tensor size or the function " +
                miopen::get_type_name<F>() + " can not be called.");
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

    template <class Integer, Integer N>
    const T& operator()(const std::array<Integer, N>& multi_id) const
    {
        auto f = [&](auto... is) { return this->desc.GetIndex(is...); };
        assert(miopen::unpack(f, multi_id) < data.size());
        return this->data[miopen::unpack(f, multi_id)];
    }

    T& operator[](std::size_t i) { return data.at(i); }

    const T& operator[](std::size_t i) const { return data.at(i); }

    typename std::vector<T>::iterator begin() { return data.begin(); }

    typename std::vector<T>::iterator end() { return data.end(); }

    typename std::vector<T>::const_iterator begin() const { return data.begin(); }

    typename std::vector<T>::const_iterator end() const { return data.end(); }

    friend std::ostream& operator<<(std::ostream& stream, const tensor& t)
    {
        return stream << t.desc;
    }
};

template <class T>
void serialize(std::istream& s, tensor<T>& x)
{
    std::vector<std::size_t> lens;
    serialize(s, lens);
    std::vector<std::size_t> strides;
    serialize(s, strides);
    x.desc = miopen::TensorDescriptor{miopen_type<T>{}, lens, strides};
    serialize(s, x.data);
}

template <class T>
void serialize(std::ostream& s, const tensor<T>& x)
{
    std::vector<std::size_t> lens    = x.desc.GetLengths();
    std::vector<std::size_t> strides = x.desc.GetStrides();
    serialize(s, lens);
    serialize(s, strides);
    serialize(s, x.data);
}

template <class T>
void save_tensor(tensor<T> input, std::string fn, int slice)
{
    std::ofstream myfile;
    myfile.open(fn, std::ofstream::out | std::ofstream::trunc);
    int b, c, h, w;
    std::tie(b, c, h, w) = miopen::tien<4>(input.desc.GetLengths());
    for(auto hidx = 0; hidx < h; hidx++)
    {
        for(auto widx = 0; widx < w; widx++)
        {
            myfile << std::setprecision(9) << input(0, slice, hidx, widx) << ", ";
        }
        myfile << std::endl;
    }
    myfile.close();
}

template <class T, class G>
tensor<T> make_tensor(std::initializer_list<std::size_t> dims, G g)
{
    // TODO: Compute float
    return tensor<T>{miopen::TensorDescriptor{miopen_type<T>{}, dims}}.generate(g);
}

// This is needed since there is no TensorDescriptor(miopenDataType_t t, const size_t* plens, int
// size) constructor
template <class T>
tensor<T> make_tensor(const std::vector<std::size_t>& dims)
{
    std::vector<int> tmpDims;

    tmpDims.resize(dims.size());

    for(int i = 0; i < tmpDims.size(); i++)
        tmpDims[i] = static_cast<int>(dims[i]);

    return tensor<T>{miopen::TensorDescriptor{
        miopen_type<T>{}, tmpDims.data(), static_cast<int>(tmpDims.size())}};
};

template <class T, class X>
tensor<T> make_tensor(const std::vector<X>& dims)
{
    // TODO: Compute float
    return tensor<T>{
        miopen::TensorDescriptor{miopen_type<T>{}, dims.data(), static_cast<int>(dims.size())}};
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
    miopen::each_args(
        std::bind(cross_args_apply{}, protect_void(std::move(f)), std::placeholders::_1, xs...),
        xs...);
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
