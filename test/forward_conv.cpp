#include <MLOpen.h>
#include "test.hpp"
#include <vector>
#include <array>
#include <iterator>
#include <memory>
#include <utility>
#include <iostream>
#include <cmath>
#include <mlopenTensor.hpp>
#include <manage_ptr.hpp>
#include <returns.hpp>

mlopenHandle_t global_handle;
struct handle_fixture
{
    mlopenHandle_t handle;
    cl_command_queue q;
    cl_context ctx;

    handle_fixture()
    {
        // mlopenCreate(&handle);
        handle = global_handle;
        mlopenGetStream(handle, &q);
        auto status = clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
        if (status != CL_SUCCESS) throw status;
    }

    template<class T>
    cl_mem create(int sz)
    {
        cl_int status = CL_SUCCESS;
        cl_mem result = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(T)*sz, nullptr, &status);
        if (status != CL_SUCCESS) throw status;
        return result;
    }

    template<class Container>
    cl_mem write(const Container& c)
    {
        auto sz = c.size()*sizeof(typename Container::value_type); 
        cl_int status = CL_SUCCESS;
        cl_mem result = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sz, nullptr, &status);
        if (status != CL_SUCCESS) throw status;
        status = clEnqueueWriteBuffer(q, result, CL_TRUE, 0, sz, c.data(), 0, nullptr, nullptr);
        if (status != CL_SUCCESS) throw status;
        return result;
    }

    template<class Container>
    cl_mem write(const Container& c, cl_mem data, int sz)
    {
        assert(sz == c.size()*sizeof(typename Container::value_type));
        cl_int status = CL_SUCCESS;
        status = clEnqueueWriteBuffer(q, result, CL_TRUE, 0, sz, c.data(), 0, nullptr, nullptr);
        if (status != CL_SUCCESS) throw status;
        return result;
    }

    template<class T>
    std::vector<T> read(cl_mem data, int sz)
    {
        std::vector<T> result(sz);
        auto status = clEnqueueReadBuffer(q, data, CL_TRUE, 0, sizeof(T)*sz, result.data(), 0, nullptr, nullptr);
        if (status != CL_SUCCESS) throw status;
        return result;
    }

    ~handle_fixture()
    {
        // mlopenDestroy(handle);
    }
};

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
        (*this)([&](T i)
        {
            (*this)([&](Ts... is)
            {
                f(i, is...);
            }, xs...);
        }, x);
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
    MLOPEN_MANAGE_PTR(mlopenTensorDescriptor_t, mlopenDestroyTensorDescriptor) desc;
    std::vector<T> data;

    tensor(int n, int c, int h, int w)
    : desc(nullptr), data(n*c*h*w)
    {
        mlopenTensorDescriptor_t local;
        mlopenCreateTensorDescriptor(&local);
        desc.reset(local);
        mlopenSet4dTensorDescriptor(
                local,
                mlopenFloat, // TODO: Pick correct type
                n,
                c,
                h,
                w);    
    }

    mlopenTensorDescriptor_t get() const
    {
        return desc.get();
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
        mlopenGet4dTensorDescriptorLengths(desc.get(), &n, &c, &h, &w);
        ford(n, c, h, w)(std::move(f));
    }

    std::array<int, 4> get_lengths() const
    {
        int n_in, c_in, h_in, w_in;
        int nStride_in, cStride_in, hStride_in, wStride_in;
        mlopenDataType_t dt;
        mlopenGet4dTensorDescriptor(
                desc.get(),
                &dt,
                &n_in,
                &c_in,
                &h_in,
                &w_in,
                &nStride_in,
                &cStride_in,
                &hStride_in,
                &wStride_in);

        return {n_in, c_in, h_in, w_in};
    }

    int index(int n, int c, int h, int w) const
    {
        return mlopenGetTensorIndex(desc.get(), {n, c, h, w});
    }

    T& operator()(int n, int c, int h, int w)
    {
        assert(this->index(n, c, h, w) < data.size());
        return this->data[this->index(n, c, h, w)];
    }

    const T& operator()(int n, int c, int h, int w) const
    {
        assert(this->index(n, c, h, w) < data.size());
        return this->data[this->index(n, c, h, w)];
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

template<mlopenConvolutionMode_t Mode>
struct conv_filter
{
    MLOPEN_MANAGE_PTR(mlopenConvolutionDescriptor_t, mlopenDestroyConvolutionDescriptor) filter;

    conv_filter(int padh = 0, int padw = 0, int u = 1, int v = 1, int upx = 1, int upy = 1)
    {
        mlopenConvolutionDescriptor_t local;
        mlopenCreateConvolutionDescriptor(&local);
        filter.reset(local);
        mlopenInitConvolutionDescriptor(local,
                Mode,
                padh,
                padw,
                u,
                v,
                upx,
                upy);   
    }

    mlopenConvolutionDescriptor_t get() const
    {
        return this->filter.get();
    }

    template<class T>
    tensor<T> get_output(const tensor<T>& input, const tensor<T>& weights) const
    {
        int x, y, z, a;
        mlopenGetConvolutionForwardOutputDim(this->filter.get(), input.get(), weights.get(), &x, &y, &z, &a);
        return {x, y, z, a};
    }
};

template<class T, mlopenConvolutionMode_t Mode>
std::vector<T> forward_conv_cpu(const tensor<T>& input, const tensor<T>& weights, const conv_filter<Mode>& filter, int bias = 0)
{
    auto out = filter.get_output(input, weights);

    int in_n, in_c, in_h, in_w;
    std::tie(in_n, in_c, in_h, in_w) = input.get_lengths();

    int wei_h, wei_w;
    std::tie(std::ignore, std::ignore, wei_h, wei_w) = weights.get_lengths();

    int pad_w, pad_h, u, v, upx, upy;
    mlopenConvolutionMode_t mode = mlopenConvolution;
    mlopenGetConvolutionDescriptor(filter.get(), &mode, &pad_h, &pad_w, &u, &v, &upx, &upy);

    out.for_each([&](int o, int w, int i, int j)
    {
        int in_off_h = i * v;
        int in_off_w = j * u;

        T acc = 0;
        for(int k = 0; k < in_c; k++) { // in_channels (RGB)
            for(int x = 0; x < wei_h; x++) {
                int in_x = in_off_h - pad_h + x;
                if(in_x >= 0 && in_x < in_h) {
                    for(int y = 0; y < wei_w; y++) {
                        int in_y = in_off_w - pad_w + y;
                        if(in_y >= 0 && in_y < in_w) {
                            acc += input(o, k, in_x, in_y) * weights(w, k, x, y);
                        }
                    }
                }
            }
        }
        out(o, w, i, j) = acc+bias;
    });
    return out.data;
}

template<class T, mlopenConvolutionMode_t Mode>
std::vector<T> forward_conv_gpu(const tensor<T>& input, const tensor<T>& weights, const conv_filter<Mode>& filter, int bias = 0)
{
    auto out = filter.get_output(input, weights);
    handle_fixture handle;

    cl_mem in_dev = handle.write(input.data);
    cl_mem wei_dev = handle.write(weights.data);
    cl_mem out_dev = handle.create<T>(out.data.size());

    int ret_algo_count;
    mlopenConvAlgoPerf_t perf;

    int alpha = 1, beta = 1;

    mlopenFindConvolutionForwardAlgorithm(handle.handle,
        input.get(),
        in_dev,
        weights.get(),
        wei_dev,
        filter.get(),
        out.get(),
        out_dev,
        1,
        &ret_algo_count,
        &perf,
        mlopenConvolutionFastest,
        NULL,
        10);

    mlopenConvolutionForward(handle.handle,
        &alpha,
        input.get(),
        in_dev,
        weights.get(),
        wei_dev,
        filter.get(),
        mlopenConvolutionFwdAlgoDirect,
        &beta,
        out.get(),
        out_dev,
        NULL,
        0);

    out.data = handle.read<T>(out_dev, out.data.size());
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

template<class T, mlopenConvolutionMode_t Mode>
std::vector<T> verify_forward_conv(const tensor<T>& input, const tensor<T>& weights, const conv_filter<Mode>& filter, int bias = 0)
{
    auto out_cpu = forward_conv_cpu(input, weights, filter, bias);
    auto out_gpu = forward_conv_gpu(input, weights, filter, bias);
    CHECK(std::distance(out_cpu.begin(), out_cpu.end()) == std::distance(out_gpu.begin(), out_gpu.end()));
    CHECK(std::equal(out_cpu.begin(), out_cpu.end(), out_gpu.begin(), float_equal_fn{}));
    // CHECK(out_cpu == out_gpu);
}
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
    auto input = tensor<float>{5, 16, 8, 8}.generate(g);
    auto weights = tensor<float>{8, 16, 5, 5}.generate(g);
    conv_filter<mlopenConvolution> filter{};
    verify_forward_conv(input, weights, filter);
}

template<class T, class... Gs>
void verify_all(Gs... gs)
{
    ford(8,8,8,8,8,8)([&](int padh, int padw, int u, int v, int upx, int upy)
    {
        conv_filter<mlopenConvolution> filter{padh, padw, u+1, v+1, upx+1, upy+1};
        ford(8,8,8,8)([&](int x1, int y1, int x2, int y2)
        {
            if (x1 >= x2 && y1 >= y2)
            {
                tensor<T> input{5, 16, x1+1, y1+1};
                tensor<T> weights{8, 16, x2+1, y2+1};
                cross_args(
                    std::bind(verify_forward_conv_gen{},
                        std::ref(input),
                        std::placeholders::_1,
                        std::ref(weights),
                        std::placeholders::_2,
                        std::ref(filter)
                    ),
                    gs...
                );
            }
        }); 
    });
}

int main() {
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
}
