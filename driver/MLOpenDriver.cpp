#include <iostream>
#include <cstdio>
#include <MLOpen.h>
#include <CL/cl.h>
#include "mloConvHost.hpp"
#include "InputFlags.hpp"
#include <vector>

int ForwardConvOnHost(int n_in, int c_in, int h_in, int w_in,
		int nStride_in, int cStride_in, int hStride_in, int wStride_in,
		int n_wei, int c_wei, int h_wei, int w_wei,
		int nStride_wei, int cStride_wei, int hStride_wei, int wStride_wei,
		int n_out, int c_out, int h_out, int w_out,
		int nStride_out, int cStride_out, int hStride_out, int wStride_out,
		int u, int v, int pad_h, int pad_w,
		float *bot, float *wei, std::vector<float> &out) {
	
	int in_pad_w = w_in + 2*pad_w;
	int in_pad_h = h_in + 2*pad_h;
	int bias = 0;

	for(int o = 0; o < n_out; o++) { // mini-batch size
		int image_off = (pad_h == 0 && pad_w == 0) ? o*nStride_in : o * c_in * in_pad_h * in_pad_w;
		for(int w = 0; w < c_out; w++) { // out_channels (num filters)
			for(int i = 0; i < h_out; i++) { // output_height (from getforwardoutputdim())
				int in_off_h = i * v;
				for(int j = 0; j < w_out; j++) { //output_width (from getforwardoutputdim())
					float acc = 0;
					int in_off_w = j * u;
					for(int k = 0; k < c_in; k++) { // in_channels (RGB)
						int chan_off = (pad_h == 0 && pad_w == 0) ? k*cStride_in : k * in_pad_h * in_pad_w;
						for(int x = 0; x < h_wei; x++) {
							for(int y = 0; y < w_wei; y++) {
								acc +=	bot[image_off + chan_off + (in_off_h+x)*in_pad_w + in_off_w + y] * wei[w*nStride_wei + k*cStride_wei + x*hStride_wei + y];
							}
						}
					}
					acc = bias != 0 ? acc+bias : acc;
					out[o*nStride_out + w*cStride_out + i*hStride_out + j] = acc;
				}
			}
		}
	}
	return 0;
}

int main()
{

	// mlopenContext APIs
	mlopenHandle_t handle;
	mlopenCreate(&handle);
	cl_command_queue q;
	mlopenGetStream(handle, &q);

	// mlopenTensor APIs
	mlopenTensorDescriptor_t inputTensor;
	mlopenCreateTensorDescriptor(handle, &inputTensor);

	mlopenInit4dTensorDescriptor(handle,
			inputTensor,
			mlopenFloat,
			100,
			32,
			8,
			8);

	int n_in, c_in, h_in, w_in;
	int n_wei, c_wei, h_wei, w_wei;
	int n_out, c_out, h_out, w_out;

	int nStride_in, cStride_in, hStride_in, wStride_in;
	int nStride_wei, cStride_wei, hStride_wei, wStride_wei;
	int nStride_out, cStride_out, hStride_out, wStride_out;
	
	mlopenDataType_t dt;
	
	mlopenGet4dTensorDescriptor(handle,
			inputTensor,
			&dt,
			&n_in,
			&c_in,
			&h_in,
			&w_in,
			&nStride_in,
			&cStride_in,
			&hStride_in,
			&wStride_in);

	size_t sz_in = n_in*c_in*h_in*w_in;

	mlopenTensorDescriptor_t convFilter;
	mlopenCreateTensorDescriptor(handle, &convFilter);

	// weights
	mlopenInit4dTensorDescriptor(handle,
		convFilter,
		mlopenFloat,
		64,  // outputs
		32,   // inputs
		5,   // kernel size
		5);

	mlopenGet4dTensorDescriptor(handle,
			convFilter,
			&dt,
			&n_wei,
			&c_wei,
			&h_wei,
			&w_wei,
			&nStride_wei,
			&cStride_wei,
			&hStride_wei,
			&wStride_wei);

	size_t sz_wei = n_wei*c_wei*h_wei*w_wei;

	int alpha = 1, beta = 1;
#if 0
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
#endif

	// mlopenConvolution APIs
	//

	mlopenConvolutionDescriptor_t convDesc;
	mlopenCreateConvolutionDescriptor(handle, &convDesc);

	mlopenConvolutionMode_t mode = mlopenConvolution;

	mlopenInitConvolutionDescriptor(convDesc, mode,	0, 0, 1, 1,	1, 1);

	int pad_w, pad_h, u, v, upx, upy;
	mlopenGetConvolutionDescriptor(convDesc, &mode,	&pad_h, &pad_w, &u, &v,	&upx, &upy);

	int x, y, z, a;
	mlopenGetConvolutionForwardOutputDim(convDesc, inputTensor, convFilter, &x, &y, &z, &a);

	mlopenTensorDescriptor_t outputTensor;
	mlopenCreateTensorDescriptor(handle, &outputTensor);

	mlopenInit4dTensorDescriptor(handle,
		outputTensor,
		mlopenFloat,
		x,
		y,
		z,
		a);

	mlopenGet4dTensorDescriptor(handle,
			outputTensor,
			&dt,
			&n_out,
			&c_out,
			&h_out,
			&w_out,
			&nStride_out,
			&cStride_out,
			&hStride_out,
			&wStride_out);

	size_t sz_out = n_out*c_out*h_out*w_out;

	int ret_algo_count;
	mlopenConvAlgoPerf_t perf;

	cl_int status;
	
	float *in = new float[sz_in];
	float *wei = new float[sz_wei];
	std::vector<float> out(sz_out, 0);
	std::vector<float> outhost(sz_out, 0);
	std::vector<float> outhost1(sz_out, 0);

	for(int i = 0; i < sz_in; i++) {
		in[i] = rand() * (1.0 / RAND_MAX);
	}
	for (int i = 0; i < sz_wei; i++) {
		wei[i] = (double)(rand() * (1.0 / RAND_MAX) - 0.5) * 0.001;
	}

	mloConvForwarDirectOnHost<float>(0,
			w_wei,
			pad_w,
			u,
			h_wei,
			pad_h,
			v,
			n_in,
			n_wei,
			c_in, 
			h_out,
			w_out,
			nStride_out,
			cStride_out,
			hStride_out,
			w_in,
			h_in,
			nStride_in,
			cStride_in,
			hStride_in,
			hStride_wei,
			in,
			outhost.data(),
			wei);

	ForwardConvOnHost(n_in, c_in, h_in, w_in,
			nStride_in, cStride_in, hStride_in, wStride_in,
			n_wei, c_wei, h_wei, w_wei,
			nStride_wei, cStride_wei, hStride_wei, wStride_wei,
			n_out, c_out, h_out, w_out,
			nStride_out, cStride_out, hStride_out, wStride_out,
			u, v, pad_h, pad_w,
			in, wei, outhost1);

	cl_context ctx;
	clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, NULL);
	
	cl_mem in_dev = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 4*sz_in,NULL, &status);
	cl_mem wei_dev = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 4*sz_wei,NULL, NULL);
	cl_mem out_dev = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4*sz_out,NULL, NULL);
	
	status = clEnqueueWriteBuffer(q, in_dev, CL_TRUE, 0, 4*sz_in, in, 0, NULL, NULL);
	status |= clEnqueueWriteBuffer(q, wei_dev, CL_TRUE, 0, 4*sz_wei, wei, 0, NULL, NULL);
	status |= clEnqueueWriteBuffer(q, out_dev, CL_TRUE, 0, 4*sz_out, out.data(), 0, NULL, NULL);
	if(status != CL_SUCCESS) 
		printf("error\n");

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

	status |= clEnqueueReadBuffer(q, out_dev, CL_TRUE, 0, 4*sz_out, out.data(), 0, NULL, NULL);
	if(status != CL_SUCCESS) 
		printf("error\n");

	for(int i = 0; i < sz_out; i++) {
		//printf("%f\t%f\t%f\n, ", out[i], outhost[i], outhost1[i]);
		printf("%f\n, ", outhost1[i]);
	}
	
	mlopenFindConvolutionBackwardDataAlgorithm(handle,
			inputTensor,
			out_dev,
			convFilter,
			wei_dev,
			convDesc,
			outputTensor,
			in_dev,
			1,
			&ret_algo_count,
			&perf,
			mlopenConvolutionFastest,
			NULL,
			10);

	mlopenConvolutionBackwardData(handle,
			&alpha,
			inputTensor,
			out_dev,
			convFilter,
			wei_dev,
			convDesc,
			mlopenConvolutionBwdDataAlgo_0,
			&beta,
			outputTensor,
			in_dev,
			NULL,
			0);

	mlopenDestroyTensorDescriptor(outputTensor);
	mlopenDestroyTensorDescriptor(convFilter);
	mlopenDestroyTensorDescriptor(inputTensor);

	mlopenDestroyConvolutionDescriptor(convDesc);

	mlopenDestroy(handle);

	delete[] in;
	delete[] wei;

	return 0;
}
