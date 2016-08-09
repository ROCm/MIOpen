#include <mlopen.h>
#include "test.hpp"
#include <vector>
#include <array>
#include <iterator>
#include <memory>
#include <utility>
#include <iostream>
#include <cmath>
#include <mlopen/tensor.hpp>
#include <mlopen/convolution.hpp>
#include <mlopen/returns.hpp>
#include <limits>

template<class F>
struct protect_fn
{
    F f;
    protect_fn(F x) : f(std::move(x)) 
    {}

    template<class... Ts>
    auto operator()(Ts&&... xs) const MLOPEN_RETURNS
    (f(std::forward<Ts>(xs)...));
};

template<class F>
protect_fn<F> protect(F f)
{
    return {std::move(f)};
}

// Multidimensional for loop
struct ford_impl
{
    template<class F, class T>
    void operator()(F f, T x) const
    {
        for(T i=0;i<x;i++) f(i);
    }

    template<class F, class T, class... Ts>
    void operator()(F f, T x, Ts... xs) const
    {
#if (defined(__GNUC__) && !defined (__clang__) && __GNUC__ == 4 && __GNUC_MINOR__ < 9)
        // Workaround for https://gcc.gnu.org/bugzilla/show_bug.cgi?id=55914
        // This reverses the order of evaluation
        (*this)([&](Ts... is)
        {
            (*this)(std::bind(protect(f), std::placeholders::_1, is...), x);
        }, xs...);
#else
        (*this)([&](T i)
        {
            (*this)([&](Ts... is)
            {
                f(i, is...);
            }, xs...);
        }, x);
#endif

    }
};

template<class... Ts>
auto ford(Ts... xs) MLOPEN_RETURNS
(
    std::bind(ford_impl{}, std::placeholders::_1, xs...)
);


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

template<class T>
tensor<T> get_output_tensor(const mlopen::ConvolutionDescriptor& filter, const tensor<T>& input, const tensor<T>& weights)
{
    return tensor<T>{filter.GetForwardOutputTensor(input.desc, weights.desc)};
}

template<class T>
std::vector<T> forward_conv(const tensor<T>& input, const tensor<T>& weights, const mlopen::ConvolutionDescriptor& filter, int bias = 0)
{
    auto out = get_output_tensor(filter, input, weights);

    int in_n, in_c, in_h, in_w;
    std::tie(in_n, in_c, in_h, in_w) = mlopen::tie4(input.desc.GetLengths());

    int wei_h, wei_w;
    std::tie(std::ignore, std::ignore, wei_h, wei_w) = mlopen::tie4(weights.desc.GetLengths());

    out.for_each([&](int o, int w, int i, int j)
    {
        int in_off_h = i * filter.v;
        int in_off_w = j * filter.u;

        T acc = bias;
        for(int k = 0; k < in_c; k++) { // in_channels (RGB)
            for(int x = 0; x < wei_h; x++) {
                int in_x = in_off_h - filter.pad_h + x;
                if(in_x >= 0 && in_x < in_h) {
                    for(int y = 0; y < wei_w; y++) {
                        int in_y = in_off_w - filter.pad_w + y;
                        if(in_y >= 0 && in_y < in_w) {
                            acc += input(o, k, in_x, in_y) * weights(w, k, x, y);
                        }
                    }
                }
            }
        }
        out(o, w, i, j) = acc;
    });
    return out.data;
}

template<class T>
std::vector<T> forward_conv(mlopen::Context& handle, const tensor<T>& input, const tensor<T>& weights, const mlopen::ConvolutionDescriptor& filter, int bias = 0)
{
    auto out = get_output_tensor(filter, input, weights);

    auto in_dev = handle.Write(input.data);
    auto wei_dev = handle.Write(weights.data);
    auto out_dev = handle.Create<T>(out.data.size());

    int ret_algo_count;
    mlopenConvAlgoPerf_t perf;

    int alpha = 1, beta = 1;

    filter.FindConvFwdAlgorithm(handle,
        input.desc,
        in_dev.get(),
        weights.desc,
        wei_dev.get(),
        out.desc,
        out_dev.get(),
        1,
        &ret_algo_count,
        &perf,
        mlopenConvolutionFastest,
        NULL,
        10,
        0); // MD: Not performing exhaustiveSearch by default for now

    filter.ConvolutionForward(handle,
        &alpha,
        input.desc,
        in_dev.get(),
        weights.desc,
        wei_dev.get(),
        mlopenConvolutionFwdAlgoDirect,
        &beta,
        out.desc,
        out_dev.get(),
        NULL,
        0);

    out.data = handle.Read<T>(out_dev, out.data.size());
    return out.data;
}

struct float_equal_fn
{
    template<class T>
    bool operator()(T x, T y) const
    {
        return std::fabs(x - y) < std::numeric_limits<T>::epsilon() * std::max(x, y);
    }
};

template<class T>
void verify_forward_conv(mlopen::Context& handle, const tensor<T>& input, const tensor<T>& weights, const mlopen::ConvolutionDescriptor& filter, int bias = 0)
{
    auto out_cpu = forward_conv(input, weights, filter, bias);
    auto out_gpu = forward_conv(handle, input, weights, filter, bias);
    CHECK(std::distance(out_cpu.begin(), out_cpu.end()) == std::distance(out_gpu.begin(), out_gpu.end()));
    CHECK(std::equal(out_cpu.begin(), out_cpu.end(), out_gpu.begin(), float_equal_fn{}));
    // CHECK(out_cpu == out_gpu);
}

#if 0
struct verify_forward_conv_gen 
{
    template<class T, mlopenConvolutionMode_t Mode, class G1, class G2>
    std::vector<T> operator()(tensor<T>& input, G1 g1, tensor<T>& weights, G2 g2, const conv_filter<Mode>& filter, int bias = 0) const
    {
        input.generate(g1);
        weights.generate(g2);
        verify_forward_conv(input, weights, filter, bias);
    }
};

template<class F, class... Ts>
void each_args(F f, Ts&&... xs)
{
    std::initializer_list<int>{(f(std::forward<Ts>(xs)), 0)...};
}

struct cross_args_apply
{
    template<class F, class T, class... Ts>
    void operator()(F f, T&& x, Ts&&... xs) const
    {
        each_args(std::bind(f, std::forward<T>(x), std::placeholders::_1), std::forward<Ts>(xs)...);
    }
};

template<class F, class... Ts>
void cross_args(F f, Ts&&... xs)
{
    each_args(
        std::bind(cross_args_apply{}, protect(std::move(f)), std::placeholders::_1, std::forward<Ts>(xs)...),
    std::forward<Ts>(xs)...);
}

template<class T, class G>
void verify_one(G g)
{
    auto input = tensor<float>{16, 32, 8, 8}.generate(g);
    auto weights = tensor<float>{64, 32, 5, 5}.generate(g);
    conv_filter<mlopenConvolution> filter{1, 1};
    verify_forward_conv(input, weights, filter);
}

template<class T>
struct verify_both
{
    template<class G1, class G2>
    void operator()(G1 g1, G2 g2) const
    {
        ford(8,8,8,8,8,8)([&](int padh, int padw, int u, int v, int upx, int upy)
        {
            conv_filter<mlopenConvolution> filter{padh, padw, u+1, v+1, upx+1, upy+1};
            ford(8,8,8,8)([&](int x1, int y1, int x2, int y2)
            {
                if (x1 >= x2 && y1 >= y2)
                {
                    auto input = tensor<float>{5, 16, x1+1, y1+1}.generate(g1);
                    auto weights = tensor<float>{8, 16, x2+1, y2+1}.generate(g2);
                    conv_filter<mlopenConvolution> filter{};
                    verify_forward_conv(input, weights, filter);
                }
            }); 
        });
    }
};

template<class T, class... Gs>
void verify_all(Gs... gs)
{
    cross_args(verify_both<T>{}, gs...);
}
#endif

template<class T, class G>
void verify_one(mlopen::Context& handle, G g)
{
    auto input = tensor<T>{16, 32, 8, 8}.generate(g);
    auto weights = tensor<T>{64, 32, 5, 5}.generate(g);
    mlopen::ConvolutionDescriptor filter{1, 1};
    verify_forward_conv(handle, input, weights, filter);
}

int main() {
    mlopen::Context handle;
    auto g = [](int n, int c, int h, int w) { return n+c+h+w; };
    verify_one<float>(handle, g);

#if 0
    mlopenCreate(&global_handle);
    auto g0 = [](int, int, int, int) { return 0; };
    auto g1 = [](int, int, int, int) { return 1; };
    auto g_id = [](int n, int c, int h, int w) { return h == w ? 1 : 0; };
    auto g = [](int n, int c, int h, int w) { return n+c+h+w; };

#if MLOPEN_TEST_ALL
    printf("verify_all\n");
    verify_all<float>(g0,g1, g_id, g);
#else
    printf("verify_one\n");
    verify_one<float>(g);
#endif

    mlopenDestroy(global_handle);
#endif
}
