#include <iostream>
#include <cstdio>
#include <MLOpen.h>
#include <CL/cl.h>
#include <vector>
#include <array>
#include <iterator>
// #include "mloConvHost.hpp"

void failed(const char * msg, const char* file, int line)
{
    printf("FAILED: %s: %s:%i\n", msg, file, line);
    std::abort();
}

#define CHECK(...) if (!(__VA_ARGS__)) failed(#__VA_ARGS__, __FILE__, __LINE__)

struct handle_fixture
{
    mlopenHandle_t handle;
    cl_command_queue q;

    handle_fixture()
    {
        mlopenCreate(&handle);
        mlopenGetStream(handle, &q);
    }

    ~handle_fixture()
    {
        mlopenDestroy(handle);
    }
};

struct input_tensor_fixture : virtual handle_fixture
{
    mlopenTensorDescriptor_t inputTensor;

    input_tensor_fixture()
    {
        mlopenCreateTensorDescriptor(handle, &inputTensor);
        mlopenInit4dTensorDescriptor(handle,
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

    void run()
    {
        int n, c, h, w;
        int nStride, cStride, hStride, wStride;
        mlopenDataType_t dt;

        mlopenGet4dTensorDescriptor(handle,
                inputTensor,
                &dt,
                &n,
                &c,
                &h,
                &w,
                &nStride,
                &cStride,
                &hStride,
                &wStride);

        CHECK(dt == 1);
        CHECK(n == 100);
        CHECK(c == 32);
        CHECK(h == 8);
        CHECK(w == 8);
        CHECK(nStride == c * cStride);
        CHECK(cStride == h * hStride);
        CHECK(hStride == w * wStride);
        CHECK(wStride == 1);
    }
};

struct conv_filter_fixture : virtual handle_fixture
{
    mlopenTensorDescriptor_t convFilter;
    mlopenConvolutionDescriptor_t convDesc;

    static const mlopenConvolutionMode_t mode = mlopenConvolution;

    conv_filter_fixture()
    {
        mlopenCreateTensorDescriptor(handle, &convFilter);
        // weights
        mlopenInit4dTensorDescriptor(handle,
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
                2,
                2,
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
        // TODO: Update API to not require mode by pointer
        mlopenConvolutionMode_t lmode = mode;
        int pad_w, pad_h, u, v, upx, upy;
        mlopenGetConvolutionDescriptor(convDesc,
                &lmode,
                &pad_h, &pad_w, &u, &v,
                &upx, &upy);

        CHECK(mode == 0);
        CHECK(pad_h == 2);
        CHECK(pad_w == 2);
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

        mlopenCreateTensorDescriptor(handle, &outputTensor);

        mlopenInit4dTensorDescriptor(handle,
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
        CHECK(z == 8);
        CHECK(a == 8);
    }
};

struct conv_forward : output_tensor_fixture
{
    void run()
    {
        int alpha = 1, beta = 1;
        mlopenTransformTensor(handle,
                &alpha,
                inputTensor,
                NULL,
                &beta,
                convFilter,
                NULL);

        int value = 10;
        mlopenSetTensor(handle, inputTensor, NULL, &value);

        mlopenScaleTensor(handle, inputTensor, NULL, &alpha);

        // Setup OpenCL buffers

        cl_int status;
        const int sz = 1024;
        std::vector<float> a1(sz, 1.0);
        std::vector<float> b1(sz, 6.0);
        std::vector<float> c1(sz, 0.0);

        cl_context ctx;
        clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, NULL);

        cl_mem adev = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 4*sz,NULL, &status);
        CHECK(status == CL_SUCCESS);

        cl_mem bdev = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 4*sz,NULL, NULL);
        cl_mem cdev = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4*sz,NULL, NULL);

        status = clEnqueueWriteBuffer(q, adev, CL_TRUE, 0, 4*sz, a1.data(), 0, NULL, NULL);
        status |= clEnqueueWriteBuffer(q, bdev, CL_TRUE, 0, 4*sz, b1.data(), 0, NULL, NULL);
        status |= clEnqueueWriteBuffer(q, cdev, CL_TRUE, 0, 4*sz, c1.data(), 0, NULL, NULL);
        CHECK(status == CL_SUCCESS);

        int ret_algo_count;
        mlopenConvAlgoPerf_t perf;

        mlopenFindConvolutionForwardAlgorithm(handle,
            inputTensor,
            adev,
            convFilter,
            bdev,
            convDesc,
            outputTensor,
            cdev,
            1,
            &ret_algo_count,
            &perf,
            mlopenConvolutionFastest,
            NULL,
            10);

        mlopenConvolutionForward(handle,
            &alpha,
            inputTensor,
            NULL,
            convFilter,
            NULL,
            convDesc,
            mlopenConvolutionFwdAlgoDirect,
            &beta,
            outputTensor,
            NULL,
            NULL,
            0);
    }
};



int main() {
    input_tensor_fixture{}.run();
    conv_filter_fixture{}.run();
    output_tensor_fixture{}.run();
    conv_forward{}.run();
}


