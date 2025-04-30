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
#include "get_handle.hpp"
#include "test.hpp"
#include "random.hpp"
#include "workspace.hpp"
#include <array>
#include <iterator>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/tensor_extra.hpp>
#include <vector>

#include <thread>

#define WORKAROUND_SWDEV_511223 1

struct handle_fixture
{
    miopenHandle_t handle{};

    handle_fixture() { miopenCreate(&handle); }

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

    void run() const
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

    void run() const
    {
        // TODO: Update API to not require mode by pointer
        miopenConvolutionMode_t lcmode = c_mode;
        int pad_w, pad_h, stride_h, stride_w, upx, upy;
        STATUS(miopenGetConvolutionDescriptor(
            convDesc, &lcmode, &pad_h, &pad_w, &stride_h, &stride_w, &upx, &upy));

        EXPECT(lcmode == 0);
        EXPECT(pad_h == 0);
        EXPECT(pad_w == 0);
        EXPECT(stride_h == 1);
        EXPECT(stride_w == 1);
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
        float alpha = 1, beta = 0;

        int n, h, c, w;
        STATUS(miopenGet4dTensorDescriptorLengths(inputTensor, &n, &c, &h, &w));
        size_t sz_in = static_cast<size_t>(n) * c * h * w;

        STATUS(miopenGet4dTensorDescriptorLengths(convFilter, &n, &c, &h, &w));
        size_t sz_wei = static_cast<size_t>(n) * c * h * w;

        STATUS(miopenGet4dTensorDescriptorLengths(outputTensor, &n, &c, &h, &w));
        size_t sz_out = static_cast<size_t>(n) * c * h * w;

        size_t sz_fwd_workspace;
        STATUS(miopenConvolutionForwardGetWorkSpaceSize(
            handle, convFilter, inputTensor, convDesc, outputTensor, &sz_fwd_workspace));

        Workspace wspace{sz_fwd_workspace};

        std::vector<float> in(sz_in);
        std::vector<float> wei(sz_wei);
        std::vector<float> out(sz_out);

        for(size_t i = 0; i < sz_in; i++)
        {
            in[i] = prng::gen_canonical<float>();
        }
        for(size_t i = 0; i < sz_wei; i++)
        {
            wei[i] = prng::gen_A_to_B(-0.5f, 0.5f) * 0.001f;
        }

        auto& mhand = get_handle();

        auto in_dev  = mhand.Write(in);
        auto wei_dev = mhand.Write(wei);
        auto out_dev = mhand.Write(out);

        int value = 10;
        STATUS(miopenSetTensor(handle, inputTensor, in_dev.get(), &value));

        STATUS(miopenScaleTensor(handle, inputTensor, in_dev.get(), &alpha));

        float time;

        std::thread([&] {
            int ret_algo_count;
            miopenConvAlgoPerf_t perf;

#if MIOPEN_BUILD_DEV && !WORKAROUND_SWDEV_511223
            miopenHandle_t handle2{};
            STATUS(miopenCreate(&handle2));

            miopenHandle_t& used_handle = handle2;
#else
            miopenHandle_t& used_handle = handle;
#endif

            STATUS(miopenEnableProfiling(used_handle, Profile));

            STATUS(miopenFindConvolutionForwardAlgorithm(
                used_handle,
                inputTensor,
                in_dev.get(),
                convFilter,
                wei_dev.get(),
                convDesc,
                outputTensor,
                out_dev.get(),
                1,
                &ret_algo_count,
                &perf,
                wspace.ptr(),
                wspace.size(),
                0)); // MD: Not performing exhaustiveSearch by default for now

            STATUS(miopenConvolutionForward(used_handle,
                                            &alpha,
                                            inputTensor,
                                            in_dev.get(),
                                            convFilter,
                                            wei_dev.get(),
                                            convDesc,
                                            perf.fwd_algo,
                                            &beta,
                                            outputTensor,
                                            out_dev.get(),
                                            wspace.ptr(),
                                            wspace.size()));

            STATUS(miopenGetKernelTime(used_handle, &time));

#if MIOPEN_BUILD_DEV && !WORKAROUND_SWDEV_511223
            STATUS(miopenDestroy(handle2));
#endif
        }).join();

        if(Profile)
        {
            CHECK(time > 0.0);
        }
        else
        {
            CHECK(time == 0.0);
        }
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
