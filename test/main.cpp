#include <mlopen.h>
#include "test.hpp"
#include <vector>
#include <array>
#include <iterator>
#include <memory>
#include <mlopen/tensor_extra.hpp>

struct handle_fixture
{
    mlopenHandle_t handle;
#if MLOPEN_BACKEND_OPENCL
    cl_command_queue q;
#endif

    handle_fixture()
    {
        mlopenCreate(&handle);
#if MLOPEN_BACKEND_OPENCL
        mlopenGetStream(handle, &q);
#endif
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
        STATUS(mlopenCreateTensorDescriptor(&inputTensor));
        STATUS(mlopenSet4dTensorDescriptor(
                inputTensor,
                mlopenFloat,
                100,
                32,
                8,
                8));
        
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

        STATUS(mlopenGet4dTensorDescriptor(
                inputTensor,
                &dt,
                &n,
                &c,
                &h,
                &w,
                &nStride,
                &cStride,
                &hStride,
                &wStride));

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
    mlopenTensorDescriptor_t convFilter;
    mlopenConvolutionDescriptor_t convDesc;

    static const mlopenConvolutionMode_t mode = mlopenConvolution;

    conv_filter_fixture()
    {
        STATUS(mlopenCreateTensorDescriptor(&convFilter));
        // weights
        STATUS(mlopenSet4dTensorDescriptor(
            convFilter,
            mlopenFloat,
            64,  // outputs
            32,   // inputs
            5,   // kernel size
            5));
        
        STATUS(mlopenCreateConvolutionDescriptor(&convDesc));
        // convolution with padding 2
        STATUS(mlopenInitConvolutionDescriptor(convDesc,
                mode,
                0,
                0,
                1,
                1,
                1,
                1));
        
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
        STATUS(mlopenGetConvolutionDescriptor(convDesc,
                &lmode,
                &pad_h, &pad_w, &u, &v,
                &upx, &upy));

        EXPECT(mode == 0);
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
    mlopenTensorDescriptor_t outputTensor;
    output_tensor_fixture()
    {
        int x, y, z, a;
        STATUS(mlopenGetConvolutionForwardOutputDim(convDesc, inputTensor, convFilter, &x, &y, &z, &a));

        STATUS(mlopenCreateTensorDescriptor(&outputTensor));

        STATUS(mlopenSet4dTensorDescriptor(
            outputTensor,
            mlopenFloat,
            x,
            y,
            z,
            a));
    }
    ~output_tensor_fixture()
    {
        mlopenDestroyTensorDescriptor(outputTensor);
    }

    void run()
    {
        int x, y, z, a;
        STATUS(mlopenGetConvolutionForwardOutputDim(convDesc, inputTensor, convFilter, &x, &y, &z, &a));

        EXPECT(x == 100);
        EXPECT(y == 64);
        EXPECT(z == 4);
        EXPECT(a == 4);
    }
};

template<bool Profile>
struct conv_forward : output_tensor_fixture
{
    void run()
    {
        STATUS(mlopenEnableProfiling(handle, Profile));
        int alpha = 1, beta = 1;
        STATUS(mlopenTransformTensor(handle,
                &alpha,
                inputTensor,
                NULL,
                &beta,
                convFilter,
                NULL));

        // int value = 10;
        // STATUS(mlopenSetTensor(handle, inputTensor, NULL, &value));

        // STATUS(mlopenScaleTensor(handle, inputTensor, NULL, &alpha));

        // Setup OpenCL buffers

		int n, h, c, w;
		STATUS(mlopenGet4dTensorDescriptorLengths(inputTensor, &n, &c, &h, &w));
		size_t sz_in = n*c*h*w;
		
		STATUS(mlopenGet4dTensorDescriptorLengths(convFilter, &n, &c, &h, &w));
		size_t sz_wei = n*c*h*w;
		
		STATUS(mlopenGet4dTensorDescriptorLengths(outputTensor, &n, &c, &h, &w));
		size_t sz_out = n*c*h*w;

		size_t sz_fwd_workspace;
		STATUS(mlopenConvolutionForwardGetWorkSpaceSize(convFilter, outputTensor, convDesc, &sz_fwd_workspace));

        std::vector<float> in(sz_in);
        std::vector<float> wei(sz_wei);
        std::vector<float> out(sz_out);
        std::vector<float> fwd_workspace(sz_fwd_workspace/4);

        for(int i = 0; i < sz_in; i++) {
            in[i] = rand() * (1.0 / RAND_MAX);
        }
        for (int i = 0; i < sz_wei; i++) {
            wei[i] = (double)(rand() * (1.0 / RAND_MAX) - 0.5) * 0.001;
        }

#if MLOPEN_BACKEND_OPENCL

        cl_context ctx;
        clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, NULL);

        cl_int status = CL_SUCCESS;
		cl_mem in_dev = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 4*sz_in,NULL, &status);
		cl_mem wei_dev = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 4*sz_wei,NULL, NULL);
		cl_mem out_dev = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4*sz_out,NULL, NULL);
		cl_mem fwd_workspace_dev = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sz_fwd_workspace, NULL, NULL);

		status = clEnqueueWriteBuffer(q, in_dev, CL_TRUE, 0, 4*sz_in, in.data(), 0, NULL, NULL);
		status |= clEnqueueWriteBuffer(q, wei_dev, CL_TRUE, 0, 4*sz_wei, wei.data(), 0, NULL, NULL);
		status |= clEnqueueWriteBuffer(q, out_dev, CL_TRUE, 0, 4*sz_out, out.data(), 0, NULL, NULL);
		status |= clEnqueueWriteBuffer(q, fwd_workspace_dev, CL_TRUE, 0, sz_fwd_workspace, fwd_workspace.data(), 0, NULL, NULL);
		EXPECT(status == CL_SUCCESS);

#elif MLOPEN_BACKEND_HIP || MLOPEN_BACKEND_HIPOC

        void * in_dev;
        void * wei_dev;
        void * out_dev;
		void * fwd_workspace_dev;

        EXPECT(hipMalloc(&in_dev, 4*sz_in) == hipSuccess);
        EXPECT(hipMalloc(&wei_dev, 4*sz_wei) == hipSuccess);
        EXPECT(hipMalloc(&out_dev, 4*sz_out) == hipSuccess);
        EXPECT(hipMalloc(&fwd_workspace_dev, sz_fwd_workspace) == hipSuccess);

        EXPECT(hipMemcpy(in_dev, in.data(), 4*sz_in, hipMemcpyHostToDevice) == hipSuccess);
        EXPECT(hipMemcpy(wei_dev, wei.data(), 4*sz_wei, hipMemcpyHostToDevice) == hipSuccess);
        EXPECT(hipMemcpy(out_dev, out.data(), 4*sz_out, hipMemcpyHostToDevice) == hipSuccess);
        EXPECT(hipMemcpy(fwd_workspace_dev, fwd_workspace.data(), sz_fwd_workspace, hipMemcpyHostToDevice) == hipSuccess);

#endif

        int ret_algo_count;
        mlopenConvAlgoPerf_t perf;

        STATUS(mlopenFindConvolutionForwardAlgorithm(handle,
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

        STATUS(mlopenConvolutionForward(handle,
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
            fwd_workspace_dev,
            sz_fwd_workspace));

        float time;
        STATUS(mlopenGetKernelTime(handle, &time));
        if (Profile) 
        { 
            CHECK(time > 0.0);
        }
        else 
        { 
            CHECK(time == 0.0);
        }

        // Potential memory leak free memory at end of function
#if MLOPEN_BACKEND_OPENCL
		clReleaseMemObject(in_dev);
		clReleaseMemObject(wei_dev);
		clReleaseMemObject(out_dev);
		clReleaseMemObject(fwd_workspace_dev);

#elif MLOPEN_BACKEND_HIP || MLOPEN_BACKEND_HIPOC
        hipFree(in_dev);
        hipFree(wei_dev);
        hipFree(out_dev);
		hipFree(fwd_workspace_dev);
#endif
    }
};

int main() {
    run_test<input_tensor_fixture>();
    run_test<conv_filter_fixture>();
    run_test<output_tensor_fixture>();
    run_test<conv_forward<true>>();
    run_test<conv_forward<false>>();
}


