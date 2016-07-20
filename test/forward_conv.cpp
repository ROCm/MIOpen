#include <MLOpen.h>
#include "test.hpp"
#include <vector>
#include <array>
#include <iterator>
#include <memory>
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
        clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
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
        clEnqueueReadBuffer(q, data, CL_TRUE, 0, sizeof(T)*sz, result.data(), 0, NULL, NULL);
        return result;
    }

    ~handle_fixture()
    {
        mlopenDestroy(handle);
    }
};

struct input_tensor_fixture
{
    mlopenTensorDescriptor_t inputTensor;

    input_tensor_fixture()
    {
        mlopenCreateTensorDescriptor(&inputTensor);
        mlopenSet4dTensorDescriptor(
                inputTensor,
                mlopenFloat,
                100,
                32,
                8,
                8);
        
    }

    ~input_tensor_fixture()
    {
        mlopenDestroyTensorDescriptor(inputTensor);
    }
};

struct conv_filter_fixture : virtual handle_fixture
{
    mlopenTensorDescriptor_t convFilter;
    mlopenConvolutionDescriptor_t convDesc;

    static const mlopenConvolutionMode_t mode = mlopenConvolution;

    conv_filter_fixture()
    {
        mlopenCreateTensorDescriptor(&convFilter);
        // weights
        mlopenSet4dTensorDescriptor(
            convFilter,
            mlopenFloat,
            64,  // outputs
            32,   // inputs
            5,   // kernel size
            5);
        
        mlopenCreateConvolutionDescriptor(handle, &convDesc);
        // convolution with padding 2
        mlopenInitConvolutionDescriptor(convDesc,
                mode,
                0,
                0,
                1,
                1,
                1,
                1);
        
    }
    ~conv_filter_fixture()
    {
        mlopenDestroyTensorDescriptor(convFilter);
        mlopenDestroyConvolutionDescriptor(convDesc);
    }

    void run()
    {
        mlopenConvolutionMode_t lmode = mode;
        int pad_w, pad_h, u, v, upx, upy;
        mlopenGetConvolutionDescriptor(convDesc,
                &lmode,
                &pad_h, &pad_w, &u, &v,
                &upx, &upy);

        CHECK(mode == 0);
        CHECK(pad_h == 0);
        CHECK(pad_w == 0);
        CHECK(u == 1);
        CHECK(v == 1);
        CHECK(upx == 1);
        CHECK(upy == 1);
    }
};

struct output_tensor_fixture : conv_filter_fixture, input_tensor_fixture
{
    mlopenTensorDescriptor_t outputTensor;
    output_tensor_fixture()
    {
        int x, y, z, a;
        mlopenGetConvolutionForwardOutputDim(convDesc, inputTensor, convFilter, &x, &y, &z, &a);

        mlopenCreateTensorDescriptor(&outputTensor);

        mlopenSet4dTensorDescriptor(
            outputTensor,
            mlopenFloat,
            x,
            y,
            z,
            a);
    }
    ~output_tensor_fixture()
    {
        mlopenDestroyTensorDescriptor(outputTensor);
    }

    void run()
    {
        int x, y, z, a;
        mlopenGetConvolutionForwardOutputDim(convDesc, inputTensor, convFilter, &x, &y, &z, &a);

        CHECK(x == 100);
        CHECK(y == 64);
        CHECK(z == 4);
        CHECK(a == 4);
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

struct conv_forward : output_tensor_fixture
{
    void run()
    {

        // Setup OpenCL buffers

        std::vector<float> in(mlopenGetTensorDescriptorElementSize(inputTensor));
        std::vector<float> wei(mlopenGetTensorDescriptorElementSize(convFilter));

        for(auto&& x:in) {
            x = rand() * (1.0 / RAND_MAX);
        }
        for (auto&& x:wei) {
            x = (double)(rand() * (1.0 / RAND_MAX) - 0.5) * 0.001;
        }

        cl_mem in_dev = this->write(in);
        cl_mem wei_dev = this->write(wei);
        cl_mem out_dev = this->create<float>(mlopenGetTensorDescriptorElementSize(outputTensor));

        int ret_algo_count;
        mlopenConvAlgoPerf_t perf;

        int alpha = 1, beta = 1;

        mlopenFindConvolutionForwardAlgorithm(handle,
            inputTensor,
            in_dev,
            convFilter,
            wei_dev,
            convDesc,
            outputTensor,
            out_dev,
            1,
            &ret_algo_count,
            &perf,
            mlopenConvolutionFastest,
            NULL,
            10);

        mlopenConvolutionForward(handle,
            &alpha,
            inputTensor,
            in_dev,
            convFilter,
            wei_dev,
            convDesc,
            mlopenConvolutionFwdAlgoDirect,
            &beta,
            outputTensor,
            out_dev,
            NULL,
            0);

        std::vector<float> out = this->read<float>(out_dev, mlopenGetTensorDescriptorElementSize(outputTensor));

        auto host_out = ForwardConvOnHost(inputTensor,
            in,
            convFilter,
            wei,
            convDesc,
            outputTensor);

        CHECK(out == host_out);

    }
};

int main() {
    run_test<conv_forward>();
}
