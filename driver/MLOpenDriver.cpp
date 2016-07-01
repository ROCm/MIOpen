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
			19,
			3,
			32,
			32);

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
	printf("%d %d %d %d %d %d %d %d (should be 19, 3, 32, 32, 1, 1, 1, 1)\n", n, c, h, w, nStride, cStride, hStride, wStride);

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

	mlopenInitConvolutionDescriptor(convDesc,
			mode,
			1,
			1,
			1,
			1,
			1,
			1);

	int pad_w, pad_h, u, v, upx, upy;
	mlopenGetConvolutionDescriptor(convDesc,
			&mode,
			&pad_h, &pad_w, &u, &v,
			&upx, &upy);

	printf("%d %d %d %d %d %d %d (Should be 0, 1, 1, 1, 1, 1, 1)\n", mode, pad_h, pad_w, u, v, upx, upy);

	mlopenGetConvolutionForwardOutputDim(convDesc, 
			tensor,
			t1,
			&n,
			&c,
			&h,
			&w);
	
	mlopenTensorDescriptor_t t2;
	mlopenCreateTensorDescriptor(handle, &t2);
	int ret_algo_count;
	mlopenConvAlgoPerf_t perf;

	mlopenFindConvolutionForwardAlgorithm(handle,
			tensor,
			NULL,
			t1,
			NULL,
			convDesc,
			t2,
			NULL,
			1,
			&ret_algo_count,
			&perf,
			mlopenConvolutionFastest,
			NULL,
			10);

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
