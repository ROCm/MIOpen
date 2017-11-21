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
#include "test.hpp"
#include <array>
#include <iterator>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/tensor_extra.hpp>
#include <vector>

#ifdef __MINGW32__
#include <mingw.thread.h>
#else
#include <thread>
#endif

struct handle_fixture
{
    miopenHandle_t handle{};
#if MIOPEN_BACKEND_OPENCL
    cl_command_queue q{};
#endif

    handle_fixture()
    {
        miopenCreate(&handle);
#if MIOPEN_BACKEND_OPENCL
        miopenGetStream(handle, &q);
#endif
    }

    ~handle_fixture() { miopenDestroy(handle); }
};

struct input_tensor_fixture
{
    miopenTensorDescriptor_t inputTensor{};

    input_tensor_fixture()
    {
        STATUS(miopenCreateTensorDescriptor(&inputTensor));
        STATUS(miopenSet4dTensorDescriptor(inputTensor, miopenFloat, 100, 32, 8, 8));
    }

    ~input_tensor_fixture() { miopenDestroyTensorDescriptor(inputTensor); }

    void run()
    {
        int n, c, h, w;
        int nStride, cStride, hStride, wStride;
        miopenDataType_t dt;

        STATUS(miopenGet4dTensorDescriptor(
            inputTensor, &dt, &n, &c, &h, &w, &nStride, &cStride, &hStride, &wStride));

        EXPECT(dt == 1);
        EXPECT(n == 100);
        EXPECT(c == 32);
        EXPECT(h == 8);
        EXPECT(w == 8);
        EXPECT(nStride == c * cStride);
        EXPECT(cStride == h * hStride);
        EXPECT(hStride == w * wStride);
        EXPECT(wStride == 1);
    }
};

struct conv_filter_fixture : virtual handle_fixture
{
    miopenTensorDescriptor_t convFilter{};
    miopenConvolutionDescriptor_t convDesc{};

    static const miopenConvolutionMode_t c_mode = miopenConvolution;
    static const miopenPaddingMode_t p_mode     = miopenPaddingDefault;

    conv_filter_fixture()
    {
        STATUS(miopenCreateTensorDescriptor(&convFilter));
        // weights
        STATUS(miopenSet4dTensorDescriptor(convFilter,
                                           miopenFloat,
                                           64, // outputs
                                           32, // inputs
                                           5,  // kernel size
                                           5));

        STATUS(miopenCreateConvolutionDescriptor(&convDesc));
        // convolution with padding 2
        STATUS(miopenInitConvolutionDescriptor(convDesc, c_mode, 0, 0, 1, 1, 1, 1));
    }
    ~conv_filter_fixture()
    {
        miopenDestroyTensorDescriptor(convFilter);
        miopenDestroyConvolutionDescriptor(convDesc);
    }

    void run()
    {
        // TODO: Update API to not require mode by pointer
        miopenConvolutionMode_t lcmode = c_mode;
        int pad_w, pad_h, u, v, upx, upy;
        STATUS(
            miopenGetConvolutionDescriptor(convDesc, &lcmode, &pad_h, &pad_w, &u, &v, &upx, &upy));

        EXPECT(lcmode == 0);
        EXPECT(pad_h == 0);
        EXPECT(pad_w == 0);
        EXPECT(u == 1);
        EXPECT(v == 1);
        EXPECT(upx == 1);
        EXPECT(upy == 1);
    }
};

struct output_tensor_fixture : conv_filter_fixture, input_tensor_fixture
{
    miopenTensorDescriptor_t outputTensor{};
    output_tensor_fixture()
    {
        int x, y, z, a;
        STATUS(miopenGetConvolutionForwardOutputDim(
            convDesc, inputTensor, convFilter, &x, &y, &z, &a));

        STATUS(miopenCreateTensorDescriptor(&outputTensor));

        STATUS(miopenSet4dTensorDescriptor(outputTensor, miopenFloat, x, y, z, a));
    }
    ~output_tensor_fixture() { miopenDestroyTensorDescriptor(outputTensor); }

    void run()
    {
        int x, y, z, a;
        STATUS(miopenGetConvolutionForwardOutputDim(
            convDesc, inputTensor, convFilter, &x, &y, &z, &a));

        EXPECT(x == 100);
        EXPECT(y == 64);
        EXPECT(z == 4);
        EXPECT(a == 4);
    }
};

template <bool Profile>
struct conv_forward : output_tensor_fixture
{
    void run()
    {
        STATUS(miopenEnableProfiling(handle, Profile));
        float alpha = 1, beta = 0;

        // Setup OpenCL buffers

        int n, h, c, w;
        STATUS(miopenGet4dTensorDescriptorLengths(inputTensor, &n, &c, &h, &w));
        size_t sz_in = n * c * h * w;

        STATUS(miopenGet4dTensorDescriptorLengths(convFilter, &n, &c, &h, &w));
        size_t sz_wei = n * c * h * w;

        STATUS(miopenGet4dTensorDescriptorLengths(outputTensor, &n, &c, &h, &w));
        size_t sz_out = n * c * h * w;

        size_t sz_fwd_workspace;
        STATUS(miopenConvolutionForwardGetWorkSpaceSize(
            handle, convFilter, inputTensor, convDesc, outputTensor, &sz_fwd_workspace));

        std::vector<float> in(sz_in);
        std::vector<float> wei(sz_wei);
        std::vector<float> out(sz_out);
        std::vector<float> fwd_workspace(sz_fwd_workspace / 4);

        for(int i = 0; i < sz_in; i++)
        {
            in[i] = rand() * (1.0 / RAND_MAX);
        }
        for(int i = 0; i < sz_wei; i++)
        {
            wei[i] = static_cast<double>(rand() * (1.0 / RAND_MAX) - 0.5) * 0.001;
        }

#if MIOPEN_BACKEND_OPENCL

        cl_context ctx;
        clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);

        cl_int status  = CL_SUCCESS;
        cl_mem in_dev  = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 4 * sz_in, nullptr, &status);
        cl_mem wei_dev = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 4 * sz_wei, nullptr, nullptr);
        cl_mem out_dev = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4 * sz_out, nullptr, nullptr);
        cl_mem fwd_workspace_dev =
            clCreateBuffer(ctx, CL_MEM_READ_WRITE, sz_fwd_workspace, nullptr, nullptr);

        status =
            clEnqueueWriteBuffer(q, in_dev, CL_TRUE, 0, 4 * sz_in, in.data(), 0, nullptr, nullptr);
        status |= clEnqueueWriteBuffer(
            q, wei_dev, CL_TRUE, 0, 4 * sz_wei, wei.data(), 0, nullptr, nullptr);
        status |= clEnqueueWriteBuffer(
            q, out_dev, CL_TRUE, 0, 4 * sz_out, out.data(), 0, nullptr, nullptr);
        status |= clEnqueueWriteBuffer(q,
                                       fwd_workspace_dev,
                                       CL_TRUE,
                                       0,
                                       sz_fwd_workspace,
                                       fwd_workspace.data(),
                                       0,
                                       nullptr,
                                       nullptr);
        EXPECT(status == CL_SUCCESS);

#elif MIOPEN_BACKEND_HIP

        void* in_dev;
        void* wei_dev;
        void* out_dev;
        void* fwd_workspace_dev;

        EXPECT(hipMalloc(&in_dev, 4 * sz_in) == hipSuccess);
        EXPECT(hipMalloc(&wei_dev, 4 * sz_wei) == hipSuccess);
        EXPECT(hipMalloc(&out_dev, 4 * sz_out) == hipSuccess);
        EXPECT(hipMalloc(&fwd_workspace_dev, sz_fwd_workspace) == hipSuccess);

        EXPECT(hipMemcpy(in_dev, in.data(), 4 * sz_in, hipMemcpyHostToDevice) == hipSuccess);
        EXPECT(hipMemcpy(wei_dev, wei.data(), 4 * sz_wei, hipMemcpyHostToDevice) == hipSuccess);
        EXPECT(hipMemcpy(out_dev, out.data(), 4 * sz_out, hipMemcpyHostToDevice) == hipSuccess);
        EXPECT(hipMemcpy(fwd_workspace_dev,
                         fwd_workspace.data(),
                         sz_fwd_workspace,
                         hipMemcpyHostToDevice) == hipSuccess);

#endif
        int value = 10;
        STATUS(miopenSetTensor(handle, inputTensor, in_dev, &value));

        STATUS(miopenScaleTensor(handle, inputTensor, in_dev, &alpha));

        int ret_algo_count;
        miopenConvAlgoPerf_t perf;

        std::thread([&] {

            STATUS(miopenFindConvolutionForwardAlgorithm(
                handle,
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
                fwd_workspace_dev,
                sz_fwd_workspace,
                0)); // MD: Not performing exhaustiveSearch by default for now

            STATUS(miopenConvolutionForward(handle,
                                            &alpha,
                                            inputTensor,
                                            in_dev,
                                            convFilter,
                                            wei_dev,
                                            convDesc,
                                            miopenConvolutionFwdAlgoDirect,
                                            &beta,
                                            outputTensor,
                                            out_dev,
                                            fwd_workspace_dev,
                                            sz_fwd_workspace));

        }).join();

        float time;
        STATUS(miopenGetKernelTime(handle, &time));
        if(Profile)
        {
            CHECK(time > 0.0);
        }
        else
        {
            CHECK(time == 0.0);
        }

// Potential memory leak free memory at end of function
#if MIOPEN_BACKEND_OPENCL
        clReleaseMemObject(in_dev);
        clReleaseMemObject(wei_dev);
        clReleaseMemObject(out_dev);
        clReleaseMemObject(fwd_workspace_dev);

#elif MIOPEN_BACKEND_HIP
        hipFree(in_dev);
        hipFree(wei_dev);
        hipFree(out_dev);
        hipFree(fwd_workspace_dev);
#endif
    }
};

int main()
{
    run_test<input_tensor_fixture>();
    run_test<conv_filter_fixture>();
    run_test<output_tensor_fixture>();
    run_test<conv_forward<true>>();
    run_test<conv_forward<false>>();
}
