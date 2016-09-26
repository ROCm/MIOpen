#include <mlopen.h>
#include "test.hpp"
#include <vector>
#include <array>
#include <iterator>
#include <algorithm>
#include <numeric>
#include <memory>
#include <utility>
#include <iostream>
#include <cmath>
#include <mlopen/tensor.hpp>
#include <mlopen/convolution.hpp>
#include <mlopen/returns.hpp>
#include <mlopen/each_args.hpp>
#include <limits>
#include <thread>

#include "network_data.hpp"

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

template<class F>
void par_for(std::size_t n, F f)
{
    const auto threadsize = std::thread::hardware_concurrency();
    if (n < threadsize)
    {
        for(std::size_t i=0;i<n;i++) f(i);
    }
    else
    {
        std::vector<std::thread> threads(threadsize);
        const std::size_t grainsize = std::ceil(static_cast<double>(n) / threads.size());

        std::size_t work = 0;
        std::generate(threads.begin(), threads.end(), [&]
        {
            auto result = std::thread([&, work]
            {
                std::size_t start = work;
                std::size_t last = std::min(n, work+grainsize);
                for(std::size_t i=start;i<last;i++) 
                {
                    f(i);
                }
            });
            work += grainsize;
            return result;
        });
        assert(work >= n);
        // TODO: Should be in destructor
        for(auto&& t:threads)
        {
            if (t.joinable()) t.join();
        }
    }
}

// Multidimensional for loop
struct ford_impl
{
    template<class F>
    void operator()(F f) const
    {
        f();
    }

    template<class F, class T, class... Ts>
    void operator()(F f, T x, Ts... xs) const
    {
        // Workaround for https://gcc.gnu.org/bugzilla/show_bug.cgi?id=55914
        for(T i=0;i<x;i++)
        {
            (*this)([&](Ts... is)
            {
                f(i, is...);
            }, xs...);
        }
    }
};

template<class... Ts>
auto ford(Ts... xs) MLOPEN_RETURNS
(
    std::bind(ford_impl{}, std::placeholders::_1, xs...)
);

struct par_ford_impl
{
    template<class F, class... Ts>
    void operator()(F f, Ts... xs) const
    {
        using array_type = std::array<std::size_t, sizeof...(Ts)>;
        array_type lens = {{static_cast<std::size_t>(xs)...}};
        array_type strides;
        strides.fill(1);
        std::partial_sum(lens.rbegin(), lens.rend()-1, strides.rbegin()+1, std::multiplies<std::size_t>());
        auto size = std::accumulate(lens.begin(), lens.end(), 1, std::multiplies<std::size_t>());
        par_for(size, [&](std::size_t i)
        {
            array_type indices;
            std::transform(strides.begin(), strides.end(), lens.begin(), indices.begin(), [&](size_t stride, size_t len)
            {
                return (i / stride) % len;
            });
            mlopen::unpack(f, indices);
        });
    }
};

template<class... Ts>
auto par_ford(Ts... xs) MLOPEN_RETURNS
(
    std::bind(par_ford_impl{}, std::placeholders::_1, xs...)
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

    out.par_for_each([&](int o, int w, int i, int j)
    {
        const int in_off_h = i * filter.v;
        const int in_off_w = j * filter.u;

        T acc = bias;
        ford(in_c, wei_h, wei_w)([&](int k, int x, int y)
        {
            const int in_x = in_off_h - filter.pad_h + x;
            const int in_y = in_off_w - filter.pad_w + y;
            if(in_x >= 0 && in_x < in_h && in_y >= 0 && in_y < in_w) {
                acc += input(o, k, in_x, in_y) * weights(w, k, x, y);
            }
        });
        out(o, w, i, j) = acc;
    });
    return out.data;
}

template<class T>
std::vector<T> forward_conv(mlopen::Handle& handle, const tensor<T>& input, const tensor<T>& weights, const mlopen::ConvolutionDescriptor& filter, int /* bias */ = 0)
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
        return std::fabs(x - y) < std::max(std::numeric_limits<T>::epsilon() * std::max(x, y), std::numeric_limits<T>::epsilon());
    }
};

template<class R1, class R2>
bool float_equal_range(R1&& r1, R2&& r2)
{
    return std::distance(r1.begin(), r1.end()) == std::distance(r2.begin(), r2.end()) &&
        std::equal(r1.begin(), r1.end(), r2.begin(), float_equal_fn{});
}

template<class R1, class R2, class Op>
float accumulate_difference(R1&& r1, R2&& r2, Op op)
{
    return std::inner_product(r1.begin(), r1.end(), r2.begin(), 0.0, op,
        [](float x, float y) { return std::fabs(x - y); }
    );
}

template<class T>
void verify_forward_conv(mlopen::Handle& handle, const tensor<T>& input, const tensor<T>& weights, const mlopen::ConvolutionDescriptor& filter, int bias = 0)
{
    auto out_cpu = forward_conv(input, weights, filter, bias);
    auto out_gpu = forward_conv(handle, input, weights, filter, bias);
    int size = std::distance(out_cpu.begin(), out_cpu.end());
    CHECK(std::distance(out_cpu.begin(), out_cpu.end()) == std::distance(out_gpu.begin(), out_gpu.end()));
    if (!float_equal_range(out_cpu, out_gpu))
    {
        std::cout << "FAILED: " << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
        std::cout << "Weights tensor: " << weights.desc.ToString() << std::endl;

        std::cout 
            << "Average difference: " 
            << (accumulate_difference(out_cpu, out_gpu, std::plus<float>()) / size) 
            << std::endl;
        std::cout 
            << "Max difference: " 
            << (accumulate_difference(out_cpu, out_gpu, [](float x, float y) { return std::max(x, y); })) 
            << std::endl;
    }
    // CHECK(out_cpu == out_gpu);
}

struct cross_args_apply
{
    template<class F, class T, class... Ts>
    void operator()(F f, T&& x, Ts&&... xs) const
    {
        mlopen::each_args(std::bind(f, std::forward<T>(x), std::placeholders::_1), std::forward<Ts>(xs)...);
    }
};

template<class F, class... Ts>
void cross_args(F f, Ts&&... xs)
{
    mlopen::each_args(
        std::bind(cross_args_apply{}, protect(std::move(f)), std::placeholders::_1, std::forward<Ts>(xs)...),
    std::forward<Ts>(xs)...);
}

template<class T>
struct verify_both
{
    template<class G1, class G2>
    void operator()(mlopen::Handle& handle, G1 g1, G2 g2) const
    {
        visit_network<T>([&](mlopen::TensorDescriptor input_desc, mlopen::TensorDescriptor weights_desc)
        {
            auto input = tensor<T>{std::move(input_desc)}.generate(g1);
            auto weights = tensor<T>{std::move(weights_desc)}.generate(g2);
            mlopen::ConvolutionDescriptor filter{1, 1};

            verify_forward_conv(handle, input, weights, filter);
        });
    }
};

template<class T, class... Gs>
void verify_all(mlopen::Handle& handle, Gs... gs)
{
    cross_args(
        std::bind(verify_both<T>{}, std::ref(handle), std::placeholders::_1, std::placeholders::_2), 
        gs...);
}

template<class T, class G>
void verify_one(mlopen::Handle& handle, G g)
{
    auto input = tensor<T>{16, 32, 8, 8}.generate(g);
    auto weights = tensor<T>{64, 32, 5, 5}.generate(g);
    mlopen::ConvolutionDescriptor filter{1, 1};
    verify_forward_conv(handle, input, weights, filter);
}

int main() {
    mlopen::Handle handle;
    auto g0 = [](int, int, int, int) { return 0; };
    auto g1 = [](int, int, int, int) { return 1; };
    auto g_id = [](int, int, int h, int w) { return h == w ? 1 : 0; };
    auto g = [](int n, int c, int h, int w)
    {
        double x = (547*n+701*c+877*h+1049*w+173)%1223;
        return x/691.0;
    };
    (void)g0;
    (void)g1;
    (void)g_id;
    (void)g;
#if MLOPEN_TEST_ALL
    printf("verify_all\n");
    verify_all<float>(handle, g0,g1, g_id, g);
#else
    printf("verify_one\n");
    verify_one<float>(handle, g);
#endif
}
