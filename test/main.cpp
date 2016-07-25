#include <MLOpen.h>
#include "test.hpp"
#include <vector>
#include <array>
#include <iterator>
#include <memory>
#include <mlopenTensor.hpp>

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

struct input_tensor_fixture //: virtual handle_fixture
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

    void run()
    {
        int n, c, h, w;
        int nStride, cStride, hStride, wStride;
        mlopenDataType_t dt;

        mlopenGet4dTensorDescriptor(
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
        mlopenCreateTensorDescriptor(&convFilter);
        // weights
        mlopenSet4dTensorDescriptor(
            convFilter,
            mlopenFloat,
            64,  // outputs
            32,   // inputs
            5,   // kernel size
            5);
        
        mlopenCreateConvolutionDescriptor(&convDesc);
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
        // TODO: Update API to not require mode by pointer
        mlopenConvolutionMode_t lmode = mode;
        int pad_w, pad_h, u, v, upx, upy;
        mlopenGetConvolutionDescriptor(convDesc,
                &lmode,
                &pad_h, &pad_w, &u, &v,
                &upx, &upy);

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

        EXPECT(x == 100);
        EXPECT(y == 64);
        EXPECT(z == 4);
        EXPECT(a == 4);
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

		int n, h, c, w;
		mlopenGet4dTensorDescriptorLengths(inputTensor, &n, &c, &h, &w);
		size_t sz_in = n*c*h*w;
		
		mlopenGet4dTensorDescriptorLengths(convFilter, &n, &c, &h, &w);
		size_t sz_wei = n*c*h*w;
		
		mlopenGet4dTensorDescriptorLengths(outputTensor, &n, &c, &h, &w);
		size_t sz_out = n*c*h*w;

        cl_int status = CL_SUCCESS;
		float *in = new float[sz_in];
		float *wei = new float[sz_wei];
		std::vector<float> out(sz_out, 0);

		for(int i = 0; i < sz_in; i++) {
			in[i] = rand() * (1.0 / RAND_MAX);
		}
		for (int i = 0; i < sz_wei; i++) {
			wei[i] = (double)(rand() * (1.0 / RAND_MAX) - 0.5) * 0.001;
		}

        cl_context ctx;
        clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, NULL);

		cl_mem in_dev = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 4*sz_in,NULL, &status);
		cl_mem wei_dev = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 4*sz_wei,NULL, NULL);
		cl_mem out_dev = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4*sz_out,NULL, NULL);

		status = clEnqueueWriteBuffer(q, in_dev, CL_TRUE, 0, 4*sz_in, in, 0, NULL, NULL);
		status |= clEnqueueWriteBuffer(q, wei_dev, CL_TRUE, 0, 4*sz_wei, wei, 0, NULL, NULL);
		status |= clEnqueueWriteBuffer(q, out_dev, CL_TRUE, 0, 4*sz_out, out.data(), 0, NULL, NULL);
		EXPECT(status == CL_SUCCESS);

        int ret_algo_count;
        mlopenConvAlgoPerf_t perf;

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
    }
};

int main() {
    run_test<input_tensor_fixture>();
    run_test<conv_filter_fixture>();
    run_test<output_tensor_fixture>();
    run_test<conv_forward>();
}


