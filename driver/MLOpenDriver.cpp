#include <iostream>
#include <cstdio>
#include <MLOpen.h>
#include <CL/cl.h>
#include "mloConvHost.hpp"

int main()
{

	// mlopenContext APIs
	mlopenHandle_t handle;
	mlopenCreate(&handle);
	cl_command_queue q;
	mlopenGetStream(handle, &q);

	// mlopenTensor APIs
	mlopenTensorDescriptor_t tensor;
	mlopenCreateTensorDescriptor(handle, &tensor);
	mlopenInit4dTensorDescriptor(handle,
			tensor,
			mlopenFloat,
			100,
			32,
			8,
			8);

	int n, c, h, w;
	int nStride, cStride, hStride, wStride;
	mlopenDataType_t dt;

	mlopenGet4dTensorDescriptor(handle,
			tensor,
			&dt,
			&n,
			&c,
			&h,
			&w,
			&nStride,
			&cStride,
			&hStride,
			&wStride);

	std::cout<<dt<<" (shoule be 1)\n";
	printf("%d %d %d %d %d %d %d %d (should be 100, 32, 8, 8, 1, 1, 1, 1)\n", n, c, h, w, nStride, cStride, hStride, wStride);

	mlopenTensorDescriptor_t t1;
	mlopenCreateTensorDescriptor(handle, &t1);
	int alpha = 1, beta = 1;
	mlopenTransformTensor(handle,
			&alpha,
			tensor,
			NULL,
			&beta,
			t1,
			NULL);

	int value = 10;
	mlopenSetTensor(handle, tensor, NULL, &value);

	mlopenScaleTensor(handle, tensor, NULL, &alpha);

	// mlopenConvolution APIs
	//

	mlopenConvolutionDescriptor_t convDesc;
	mlopenCreateConvolutionDescriptor(handle, &convDesc);

	mlopenConvolutionMode_t mode = mlopenConvolution;

// convolution with padding 2
	mlopenInitConvolutionDescriptor(convDesc,
			mode,
			2,
			2,
			1,
			1,
			1,
			1);

	int pad_w, pad_h, u, v, upx, upy;
	mlopenGetConvolutionDescriptor(convDesc,
			&mode,
			&pad_h, &pad_w, &u, &v,
			&upx, &upy);

	printf("%d %d %d %d %d %d %d (Should be 0, 2, 2, 1, 1, 1, 1)\n", mode, pad_h, pad_w, u, v, upx, upy);



	

	mlopenTensorDescriptor_t t2;
	mlopenCreateTensorDescriptor(handle, &t2);

	mlopenInit4dTensorDescriptor(handle,
		t2,
		mlopenFloat,
		n,
		64,
		8,
		8);


	// weights
	mlopenInit4dTensorDescriptor(handle,
		t1,
		mlopenFloat,
		64,  // outputs
		32,   // inputs
		5,   // kernel size
		5);


	int ret_algo_count;
	mlopenConvAlgoPerf_t perf;

	cl_int status;
#if 1 // Test to see if we can launch the kernel and get the results back
	float *a1 = new float[1024];
	float *b1 = new float[1024];
	float *c1 = new float[1024];

	for(int i = 0; i < 1024; i++) {
		a1[i] = 1.0;
		b1[i] = 6.0;
		c1[i] = 0.0;
	}
	int sz = 1024;

	cl_context ctx;
	clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, NULL);

	cl_mem adev = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 4*sz,NULL, &status);
	if(status != CL_SUCCESS) {
		printf("error %d\n", status);
	}
	cl_mem bdev = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 4*sz,NULL, NULL);
	cl_mem cdev = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4*sz,NULL, NULL);

	status = clEnqueueWriteBuffer(q, adev, CL_TRUE, 0, 4*sz, a1, 0, NULL, NULL);
	status |= clEnqueueWriteBuffer(q, bdev, CL_TRUE, 0, 4*sz, b1, 0, NULL, NULL);
	status |= clEnqueueWriteBuffer(q, cdev, CL_TRUE, 0, 4*sz, c1, 0, NULL, NULL);
	if(status != CL_SUCCESS) 
		printf("error\n");
#endif // Test

	mlopenFindConvolutionForwardAlgorithm(handle,
			tensor,
			adev,
			t1,
			bdev,
			convDesc,
			t2,
			cdev,
			1,
			&ret_algo_count,
			&perf,
			mlopenConvolutionFastest,
			NULL,
			10);
#if 1 // Read results back
	clEnqueueReadBuffer(q, cdev, CL_TRUE, 0, 4*sz, c1, 0, NULL, NULL);

	float sum = 0.0;
	for(int i = 0; i < 1024; i++) {
		b1[i] = 6;
		sum += c1[i];
	}

	printf("\nsum %f\n, ", sum);
	sum = 0.0;

	getchar();
#endif //Results

	mlopenConvolutionForward(handle,
			&alpha,
			tensor,
			NULL,
			t1,
			NULL,
			convDesc,
			mlopenConvolutionFwdAlgoGEMM,
			&beta,
			t2,
			NULL);

	mlopenDestroyTensorDescriptor(t2);
	mlopenDestroyTensorDescriptor(t1);
	mlopenDestroyTensorDescriptor(tensor);

	mlopenDestroyConvolutionDescriptor(convDesc);

	mlopenDestroy(handle);
	return 0;
}
