#include <MLOpen.h>
#include "test.hpp"
#include <vector>
#include <array>
#include <iterator>
#include <memory>
#include <utility>
#include <iostream>
#include <mlopenTensor.hpp>

struct handle_fixture
{
    mlopenHandle_t handle;
    cl_command_queue q;
    cl_context ctx;

    handle_fixture()
    {
        mlopenCreate(&handle);
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
        mlopenDestroy(handle);
    }
};

template<class F>
void for4(int n, int c, int h, int w, F f)
{
    for(int i=0;i<n;i++)
            for(int j=0;j<c;j++)
                for(int k=0;k<h;k++)
                    for(int m=0;m<w;m++)
                        f(i, j, k, m);
}

template<class T>
struct input_tensor
{
    mlopenTensorDescriptor_t desc;
    std::vector<T> data;

    template<class Generate>
    input_tensor(Generate g, int n, int c, int h, int w)
    : data(n*c*h*w)
    {
        mlopenCreateTensorDescriptor(&desc);
        mlopenSet4dTensorDescriptor(
                desc,
                mlopenFloat, // TODO: Pick correct type
                n,
                c,
                h,
                w);

        auto iterator = data.begin();
        for4(n, c, h, w, [&](int i, int j, int k, int m)
        {
            assert(iterator < data.end());
            *iterator = g(i, j, k, m);
            ++iterator;
        });        
    }

    ~input_tensor()
    {
        mlopenDestroyTensorDescriptor(desc);
    }
};

template<class T>
struct conv_filter
{
    mlopenTensorDescriptor_t desc;
    mlopenConvolutionDescriptor_t filter;
    std::vector<T> data;

    static const mlopenConvolutionMode_t mode = mlopenConvolution;

    template<class Generate>
    conv_filter(Generate g, int n, int c, int h, int w, int padh = 0, int padw = 0, int u = 1, int v = 1, int upx = 1, int upy = 1)
    : data(n*c*h*w)
    {
        mlopenCreateTensorDescriptor(&desc);
        // weights
        mlopenSet4dTensorDescriptor(
            desc,
            mlopenFloat, // TODO: Pick correct type
            n,  // outputs
            c,   // inputs
            h,   // kernel size
            w);
        
        mlopenCreateConvolutionDescriptor(&filter);
        // convolution with padding 2
        mlopenInitConvolutionDescriptor(filter,
                mode,
                padh,
                padw,
                u,
                v,
                upx,
                upy);

        auto iterator = data.begin();
        for4(n, c, h, w, [&](int i, int j, int k, int m)
        {
            assert(iterator < data.end());
            *iterator = g(i, j, k, m);
            ++iterator;
        }); 
        
    }
    ~conv_filter()
    {
        mlopenDestroyTensorDescriptor(desc);
        mlopenDestroyConvolutionDescriptor(filter);
    }
};

template<class T>
struct output_tensor
{
    mlopenTensorDescriptor_t desc;
    std::vector<T> data;

    output_tensor(const conv_filter<T>& filter, const input_tensor<T>& input)
    {
        int x, y, z, a;
        mlopenGetConvolutionForwardOutputDim(filter.filter, input.desc, filter.desc, &x, &y, &z, &a);

        data.resize(x*y*z*a);

        mlopenCreateTensorDescriptor(&desc);

        mlopenSet4dTensorDescriptor(
            desc,
            mlopenFloat,
            x,
            y,
            z,
            a);
    }
    ~output_tensor()
    {
        mlopenDestroyTensorDescriptor(desc);
    }
};

template<class T>
std::vector<T> ForwardConvOnHost(
        const mlopenTensorDescriptor_t      xDesc,
        const std::vector<T>&               xData,
        const mlopenTensorDescriptor_t      wDesc,
        const std::vector<T>&               wData,
        const mlopenConvolutionDescriptor_t convDesc,
        const mlopenTensorDescriptor_t      yDesc)
{
    int bias = 0;
    std::vector<T> out(mlopenGetTensorDescriptorElementSize(yDesc), bias);

    int n_in, c_in, h_in, w_in;
    int nStride_in, cStride_in, hStride_in, wStride_in;
    mlopenDataType_t dt;
    mlopenGet4dTensorDescriptor(
            xDesc,
            &dt,
            &n_in,
            &c_in,
            &h_in,
            &w_in,
            &nStride_in,
            &cStride_in,
            &hStride_in,
            &wStride_in);

    int pad_w, pad_h, u, v, upx, upy;
    mlopenConvolutionMode_t mode = mlopenConvolution;
    mlopenGetConvolutionDescriptor(convDesc, &mode, &pad_h, &pad_w, &u, &v, &upx, &upy);

    int n_out, c_out, h_out, w_out;
    int nStride_out, cStride_out, hStride_out, wStride_out;
    mlopenGet4dTensorDescriptor(
            yDesc,
            &dt,
            &n_out,
            &c_out,
            &h_out,
            &w_out,
            &nStride_out,
            &cStride_out,
            &hStride_out,
            &wStride_out);

    int n_wei, c_wei, h_wei, w_wei;
    int nStride_wei, cStride_wei, hStride_wei, wStride_wei;
    mlopenGet4dTensorDescriptor(
            wDesc,
            &dt,
            &n_wei,
            &c_wei,
            &h_wei,
            &w_wei,
            &nStride_wei,
            &cStride_wei,
            &hStride_wei,
            &wStride_wei);

    int in_pad_w = w_in + 2*pad_w;
    int in_pad_h = h_in + 2*pad_h;

    for(int o = 0; o < n_out; o++) { // mini-batch size
        int image_off = (pad_h == 0 && pad_w == 0) ? o*nStride_in : o * c_in * in_pad_h * in_pad_w;
        for(int w = 0; w < c_out; w++) { // out_channels (num filters)
            for(int i = 0; i < h_out; i++) { // output_height (from getforwardoutputdim())
                int in_off_h = i * v;
                for(int j = 0; j < w_out; j++) { //output_width (from getforwardoutputdim())
                    int in_off_w = j * u;
                    for(int k = 0; k < c_in; k++) { // in_channels (RGB)
                        int chan_off = (pad_h == 0 && pad_w == 0) ? k*cStride_in : k * in_pad_h * in_pad_w;
                        for(int x = 0; x < h_wei; x++) {
                            for(int y = 0; y < w_wei; y++) {
                                out[mlopenGetTensorIndex(yDesc, {o, w, i, j})] += 
                                    xData[image_off + chan_off + (in_off_h+x)*in_pad_w + in_off_w + y] * 
                                    wData[mlopenGetTensorIndex(wDesc, {w, k, x, y})];
                            }
                        }
                    }
                }
            }
        }
    }
    return out;
}

template<class T>
void verify_forward_conv(const input_tensor<T>& input, const conv_filter<T>& filter)
{
    output_tensor<T> out{filter, input};
    handle_fixture handle;

    cl_mem in_dev = handle.write(input.data);
    cl_mem wei_dev = handle.write(filter.data);
    cl_mem out_dev = handle.create<float>(out.data.size());

    int ret_algo_count;
    mlopenConvAlgoPerf_t perf;

    int alpha = 1, beta = 1;

    mlopenFindConvolutionForwardAlgorithm(handle.handle,
        input.desc,
        in_dev,
        filter.desc,
        wei_dev,
        filter.filter,
        out.desc,
        out_dev,
        1,
        &ret_algo_count,
        &perf,
        mlopenConvolutionFastest,
        NULL,
        10);

    mlopenConvolutionForward(handle.handle,
        &alpha,
        input.desc,
        in_dev,
        filter.desc,
        wei_dev,
        filter.filter,
        mlopenConvolutionFwdAlgoDirect,
        &beta,
        out.desc,
        out_dev,
        NULL,
        0);

    out.data = handle.read<T>(out_dev, out.data.size());

    auto host_out = ForwardConvOnHost(input.desc,
        input.data,
        filter.desc,
        filter.data,
        filter.filter,
        out.desc);

    CHECK(out.data == host_out);
}

template<class F, class... Ts>
void each_args(F f, Ts&&... xs)
{
    std::initializer_list<int>{(f(std::forward<Ts>(xs)), 0)...};
}

struct swallow
{
    template<class... Ts>
    swallow(Ts&&...) {}
};

template<class T, class Input, class Filter>
void verify(int n, std::pair<int, int> ind, std::pair<int, int> fd, Input in, Filter f)
{
    verify_forward_conv<T>(
        input_tensor<T>(in, 10, 16, ind.first, ind.second),
        conv_filter<T>(f, n, 16, fd.first, fd.second)
    );
}

template<class T, class... Gs>
void verify_each_gen(int n, std::pair<int, int> ind, std::pair<int, int> fd, Gs... gs)
{
    each_args([&](auto g1)
    {
        each_args([&](auto g2)
        {
            verify<T>(n, ind, fd, g1, g2);
        }, gs...);
    }, gs...);
}

int main() {
    auto g0 = [](int, int, int, int) { return 0; };
    auto g1 = [](int, int, int, int) { return 1; };
    auto g_id = [](int n, int c, int h, int w) { return h == w ? 1 : 0; };
    auto g = [](int n, int c, int h, int w) { return n+c+h+w; };

    verify_each_gen<float>(64, std::make_pair(8, 8), std::make_pair(5, 5), g0, g1, g, g_id);

    // verify_sizes<float>(7, g, g);

    // verify_forward_conv<float>(
    //     input_tensor<float>(g, 10, 32, 8, 8),
    //     conv_filter<float>(g, 64, 32, 5, 5)
    // );
}
