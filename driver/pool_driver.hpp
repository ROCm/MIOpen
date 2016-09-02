#ifndef GUARD_MLOPEN_POOL_DRIVER_HPP
#define GUARD_MLOPEN_POOL_DRIVER_HPP

#include <cstdlib>
#include <mlopen.h>
#include <CL/cl.h>
#include "driver.hpp"
#include "mloPoolingHost.hpp"
#include "InputFlags.hpp"
#include "tensor_driver.hpp"
#include <mlopen/tensor.hpp>
#include <vector>
#include <algorithm>
#include <float.h>
#include <memory>
#include <numeric>

template<typename T>
class PoolDriver : public Driver 
{
	public:
	PoolDriver() : Driver() {
		mlopenCreateTensorDescriptor(&inputTensor);
		mlopenCreateTensorDescriptor(&outputTensor);

		mlopenCreateTensorDescriptor(&dInputTensor);
		mlopenCreateTensorDescriptor(&dOutputTensor);

		mlopenCreatePoolingDescriptor(&poolDesc);
	}

	int AddCmdLineArgs();
	int ParseCmdLineArgs(int argc, char *argv[]);
	InputFlags & GetInputFlags() { return inflags; }

	int GetandSetData();
	std::vector<int> GetInputTensorLengthsFromCmdLine();

	int SetPoolDescriptorFromCmdLineArgs();

	std::vector<int> GetOutputTensorLengths();

	int AllocateBuffersAndCopy();
	
	int RunForwardGPU();
	//int RunForwardCPU(); // Verify implements it
	
	int RunBackwardGPU();
	//int RunBackwardCPU(); // Verify implements it
	
	int VerifyBackward();
	int VerifyForward();
	~PoolDriver() {

		mlopenDestroyTensorDescriptor(outputTensor);
		mlopenDestroyTensorDescriptor(inputTensor);

		mlopenDestroyPoolingDescriptor(poolDesc);

	}
		
	private:
	InputFlags inflags;

	mlopenTensorDescriptor_t inputTensor;
	mlopenTensorDescriptor_t outputTensor;

	std::unique_ptr<GPUMem> in_dev;
	std::unique_ptr<GPUMem> out_dev;
	std::unique_ptr<GPUMem> mask_dev;

	std::vector<T> in;
	std::vector<T> out;
	std::vector<size_t> mask;
	std::vector<T> outhost;

	mlopenPoolingDescriptor_t poolDesc;
	bool do_backward;

	mlopenTensorDescriptor_t dInputTensor;
	mlopenTensorDescriptor_t dOutputTensor;

	std::unique_ptr<GPUMem> din_dev;
	std::unique_ptr<GPUMem> dout_dev;

	std::vector<T> din;
	std::vector<T> dout;
	std::vector<T> dinhost;

};

template<typename T>
int PoolDriver<T>::ParseCmdLineArgs(int argc, char *argv[]) { 
	inflags.Parse(argc, argv); 

	do_backward = !(inflags.GetValueInt("forw"));

	if(inflags.GetValueInt("time") == 1) {
		mlopenEnableProfiling(GetHandle(), true);
	}
	return 0; 
}

template<typename T>
int PoolDriver<T>::GetandSetData() {
	std::vector<int> in_len = GetInputTensorLengthsFromCmdLine();

	SetTensor4d(inputTensor, in_len);
	SetTensor4d(dInputTensor, in_len);
	SetPoolDescriptorFromCmdLineArgs();

	std::vector<int> out_len = GetOutputTensorLengths();
	SetTensor4d(outputTensor, out_len);
	SetTensor4d(dOutputTensor, out_len);
	return(0);
}

template<typename T>
int PoolDriver<T>::AddCmdLineArgs() {
	inflags.AddInputFlag("forw", 'F', "0", "Run only Forward Pooling (Default=0)", "int");
	inflags.AddInputFlag("batchsize", 'n', "100", "Mini-batch size (Default=100)", "int");
	inflags.AddInputFlag("in_channels", 'c', "3", "Number of Input Channels (Default=3)", "int");
	inflags.AddInputFlag("in_h", 'H', "32", "Input Height (Default=32)", "int");
	inflags.AddInputFlag("in_w", 'W', "32", "Input Width (Default=32)", "int");
	inflags.AddInputFlag("out_channels", 'k', "32", "Number of Output Channels (Default=32)", "int");
	inflags.AddInputFlag("win_h", 'x', "3", "Window Height (Default=3)", "int");
	inflags.AddInputFlag("win_w", 'y', "3", "Window Width (Default=3)", "int");
	inflags.AddInputFlag("pool_stride_0", 'u', "1", "Pooling Stride Vertical (Default=1)", "int");
	inflags.AddInputFlag("pool_stride_1", 'v', "1", "Pooling Stride Horizontal (Default=1)", "int");
	inflags.AddInputFlag("pad_h", 'p', "0", "Zero Padding Height (Default=0)", "int");
	inflags.AddInputFlag("pad_w", 'q', "0", "Zero Padding Width (Default=0)", "int");
	inflags.AddInputFlag("pad_val", 'r', "0", "Padding Value (Default=0)", "int");
	inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
	inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
	inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
	//inflags.AddInputFlag("back", 'b', "0", "Do Backward Pooling (Default=0)", "int");
	inflags.AddInputFlag("print", 'P', "1", "Print Pooling Dimensions (Default=1)", "int");
	inflags.AddInputFlag("mode", 'm', "max", "Pooling Mode (max, avg) (Default=max)", "str");

	return 0;
}

template<typename T>
std::vector<int> PoolDriver<T>::GetInputTensorLengthsFromCmdLine() {
	int in_n = inflags.GetValueInt("batchsize");
	int in_c = inflags.GetValueInt("in_channels");
	int in_h = inflags.GetValueInt("in_h");
	int in_w = inflags.GetValueInt("in_w");

	return std::vector<int> ({in_n, in_c, in_h, in_w});
}

template<typename T>
int PoolDriver<T>::SetPoolDescriptorFromCmdLineArgs() {

	mlopenPoolingMode_t mode;
	int pad_h = inflags.GetValueInt("pad_h");
	int pad_w = inflags.GetValueInt("pad_w");
	int u = inflags.GetValueInt("pool_stride_0");
	int v = inflags.GetValueInt("pool_stride_1");
	int win_h = inflags.GetValueInt("win_h");
	int win_w = inflags.GetValueInt("win_w");
	if((inflags.GetValueStr("mode")) == "max") {
		mode = mlopenPoolingMax;
	}
	else if((inflags.GetValueStr("mode")) == "avg") {
		mode = mlopenPoolingAverage;
	}
	else {
		printf("Incorrect Pooling Mode\n");
		exit(0);
	}

	return mlopenSet2dPoolingDescriptor(poolDesc, mode,	win_h, win_w, pad_h, pad_w, u, v);
}

template<typename T>
std::vector<int> PoolDriver<T>::GetOutputTensorLengths() {
	int n, c, h, w;

	mlopenGetPoolingForwardOutputDim(poolDesc,
			inputTensor,
			&n, &c, &h, &w);

	return std::vector<int> ({n, c, h, w});
}

template<typename T>
int PoolDriver<T>::AllocateBuffersAndCopy() {
	
	size_t in_sz = GetTensorSize(inputTensor); 
	size_t out_sz = GetTensorSize(outputTensor); 

	cl_context ctx;

	clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, NULL);

	in_dev = std::unique_ptr<GPUMem>( new GPUMem(ctx, in_sz, sizeof(float)));
	out_dev = std::unique_ptr<GPUMem> (new GPUMem(ctx, out_sz, sizeof(float)));
	mask_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(int)));
	
	din_dev = std::unique_ptr<GPUMem>( new GPUMem(ctx, in_sz, sizeof(float)));
	dout_dev = std::unique_ptr<GPUMem> (new GPUMem(ctx, out_sz, sizeof(float)));

	in = std::vector<T>(in_sz);
	out = std::vector<T>(out_sz, 0);
	mask = std::vector<size_t>(out_sz, 0);
	outhost = std::vector<T>(out_sz, 0);
	
	din = std::vector<T>(in_sz);
	dout = std::vector<T>(out_sz, 0);
	dinhost = std::vector<T>(in_sz, 0);

	for(int i = 0; i < in_sz; i++) {
		in[i] = rand() * (1.0 / RAND_MAX);
	}
	
	for (int i = 0; i < out_sz; i++) {
		dout[i] = (double)(rand() * (1.0 / RAND_MAX) - 0.5) * 0.001;
	}

	cl_int status;
	status = in_dev->ToGPU(q, in.data());
	status |= out_dev->ToGPU(q, out.data());
	
	status = din_dev->ToGPU(q, din.data());
	status |= dout_dev->ToGPU(q, dout.data());

	if(status != CL_SUCCESS) 
		printf("Error copying data to GPU\n");

	return mlopenStatusSuccess;
}

template<typename T>
int PoolDriver<T>::RunForwardGPU() {

	int alpha = 1, beta = 1;

	mlopenPoolingForward(GetHandle(),
			poolDesc,
			&alpha,
			inputTensor,
			in_dev->GetMem(),
			&beta,
			outputTensor,
			out_dev->GetMem(),
			mask_dev->GetMem(),
			do_backward,
			NULL,
			0);

	if(inflags.GetValueInt("time") == 1) {
		float time = 0.0;
		mlopenGetKernelTime(GetHandle(), &time);
		printf("GPU Kernel Time Forward Pooling Elapsed: %f ms\n", time);
	}

	out_dev->FromGPU(GetStream(), out.data());

	return mlopenStatusSuccess;
}

template<typename T>
int PoolDriver<T>::RunBackwardGPU() {

	int alpha = 1, beta = 1;

	mlopenPoolingBackward(GetHandle(),
			poolDesc,
			&alpha,
			outputTensor,
			out_dev->GetMem(),
			dOutputTensor,
			dout_dev->GetMem(),
			inputTensor,
			in_dev->GetMem(),
			&beta,
			dInputTensor,
			din_dev->GetMem(),
			NULL);

	if(inflags.GetValueInt("time") == 1) {
		float time = 0.0;
		mlopenGetKernelTime(GetHandle(), &time);
		printf("GPU Kernel Time Backward Pooling Elapsed: %f ms\n", time);
	}

	din_dev->FromGPU(GetStream(), din.data());

	return mlopenStatusSuccess;
}

template<typename T>
int PoolDriver<T>::VerifyForward() {

	int nInStride, cInStride, hInStride, wInStride;
	mlopenGet4dTensorDescriptorStrides(inputTensor, &nInStride, &cInStride, &hInStride, &wInStride);

	int nIn, cIn, hIn, wIn;
	mlopenGet4dTensorDescriptorLengths(inputTensor, &nIn, &cIn, &hIn, &wIn);

	int nOutStride, cOutStride, hOutStride, wOutStride;
	mlopenGet4dTensorDescriptorStrides(outputTensor, &nOutStride, &cOutStride, &hOutStride, &wOutStride);

	int nOut, cOut, hOut, wOut;
	mlopenGet4dTensorDescriptorLengths(outputTensor, &nOut, &cOut, &hOut, &wOut);

	mlopenPoolingMode_t	mode;
	int	windowHeight;
	int	windowWidth;
	int	pad_h;
	int	pad_w;
	int	u;
	int	v;
	mlopenGet2dPoolingDescriptor(poolDesc, &mode, &windowHeight, &windowWidth, &pad_h, &pad_w, &u, &v);

	int pooling_method = (mode == mlopenPoolingMax) ? MLO_POOLING_OP_MAX : MLO_POOLING_OP_AVE;

	bool match = mloPoolingForwardRunHostAndVerify<float>(
			pooling_method,
			pad_h,
			u,
			windowHeight,
			pad_w,
			v,
			windowWidth,
			nIn,
			cOut,
			hIn,
			wIn,
			hInStride,
			cInStride,
			nInStride,
			hOut,
			wOut,
			hOutStride,
			cOutStride,
			nOutStride,
			in.data(),
			out.data(),
			mask.data(),
			(1 << 2)
				);

	printf(match ? "Forward Pooling Verifies on CPU and GPU\n" : "Forward Pooling Verification Failed !!\n");

	return 0;
}

template<typename T>
int PoolDriver<T>::VerifyBackward() {

	int nInStride, cInStride, hInStride, wInStride;
	mlopenGet4dTensorDescriptorStrides(inputTensor, &nInStride, &cInStride, &hInStride, &wInStride);
	int nIn, cIn, hIn, wIn;
	mlopenGet4dTensorDescriptorLengths(inputTensor, &nIn, &cIn, &hIn, &wIn);
	int nOutStride, cOutStride, hOutStride, wOutStride;
	mlopenGet4dTensorDescriptorStrides(outputTensor, &nOutStride, &cOutStride, &hOutStride, &wOutStride);
	int nOut, cOut, hOut, wOut;
	mlopenGet4dTensorDescriptorLengths(outputTensor, &nOut, &cOut, &hOut, &wOut);

	int ndInStride, cdInStride, hdInStride, wdInStride;
	mlopenGet4dTensorDescriptorStrides(dInputTensor, &ndInStride, &cdInStride, &hdInStride, &wdInStride);
	int ndIn, cdIn, hdIn, wdIn;
	mlopenGet4dTensorDescriptorLengths(dInputTensor, &ndIn, &cdIn, &hdIn, &wdIn);
	int ndOutStride, cdOutStride, hdOutStride, wdOutStride;
	mlopenGet4dTensorDescriptorStrides(dOutputTensor, &ndOutStride, &cdOutStride, &hdOutStride, &wdOutStride);
	int ndOut, cdOut, hdOut, wdOut;
	mlopenGet4dTensorDescriptorLengths(dOutputTensor, &ndOut, &cdOut, &hdOut, &wdOut);


	mlopenPoolingMode_t	mode;
	int	windowHeight;
	int	windowWidth;
	int	pad_h;
	int	pad_w;
	int	u;
	int	v;
	mlopenGet2dPoolingDescriptor(poolDesc, &mode, &windowHeight, &windowWidth, &pad_h, &pad_w, &u, &v);

	int pooling_method = (mode == mlopenPoolingMax) ? MLO_POOLING_OP_MAX : MLO_POOLING_OP_AVE;

	mloPoolingBackwardRunHost<float>(
			pooling_method,
			pad_h,
			u,
			windowHeight,
			pad_w,
			v,
			windowWidth,
			// host output
			dinhost.data(),
			dout.data(),
			mask.data(),
			ndInStride,
			cdInStride,
			hdInStride,
			nInStride,
			cInStride,
			hInStride,
			wIn,
			hIn,
			cOut,
			nOut,
			ndOutStride,
			cdOutStride,
			hdOutStride,
			wOut,
			hOut,
			nOutStride,
			cOutStride,
			hOutStride
				);

	for (int i = 0; i < din.size(); i++) {
		T res_gpu = din[i];
		T res_cpu = dinhost[i];
		T diff = std::fabs(res_gpu - res_cpu);
		if (diff > std::fabs((std::max(res_gpu, res_cpu)) * std::numeric_limits<T>::epsilon())) {
			printf("Output Mismatch at: %d diff: %.10f gpu: %.10f cpu: %.10f \nBackward Pooling Verification Failed !!\n", i, diff, res_gpu, res_cpu);
			return -1;
		}
	}

	printf("Backward Pooling Verifies on CPU and GPU\n");

	return 0;
}
#endif //GUARD_MLOPEN_POOL_DRIVER_HPP
