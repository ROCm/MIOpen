#ifndef GUARD_TENSOR_HOLDER_HPP
#define GUARD_TENSOR_HOLDER_HPP

#include <mlopen/tensor.hpp>
#include "ford.hpp"

template<class T>
struct tensor
{
    mlopen::TensorDescriptor desc;
    std::vector<T> data;

    tensor(int n, int c, int h, int w)
    : desc(mlopenFloat, {n,c,h,w}), data(n*c*h*w)
    {}

    tensor(mlopen::TensorDescriptor rhs)
    : desc(std::move(rhs))
    {
        data.resize(desc.GetElementSize());
    }

    template<class G>
    tensor& generate(G g) &
    {
        this->generate_impl(g);
        return *this;
    }

    template<class G>
    tensor&& generate(G g) &&
    {
        this->generate_impl(g);
        return std::move(*this);
    }

    template<class G>
    void generate_impl(G g)
    {
        auto iterator = data.begin();
        this->for_each([&](int i, int j, int k, int m)
        {
            assert(iterator < data.end());
            *iterator = g(i, j, k, m);
            ++iterator;
        });
    }

    template<class F>
    void for_each(F f) const
    {
        int n, c, h, w;
        std::tie(n, c, h, w) = mlopen::tie4(desc.GetLengths());
        ford(n, c, h, w)(std::move(f));
    }

    template<class F>
    void par_for_each(F f) const
    {
        int n, c, h, w;
        std::tie(n, c, h, w) = mlopen::tie4(desc.GetLengths());
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
};

struct tensor_generate
{
    template<class Tensor, class G>
    Tensor&& operator()(Tensor&& t, G g) const
    {
        return std::forward<Tensor>(t.generate(g));
    }
};

#endif
