#ifndef GUARD_MLOPEN_SOFTMAX_DRIVER_HPP
#define GUARD_MLOPEN_SOFTMAX_DRIVER_HPP

#include <cstdlib>
#include <mlopen.h>
#include <CL/cl.h>
#include "driver.hpp"
#include "InputFlags.hpp"
#include "tensor_driver.hpp"
#include <mlopen/tensor.hpp>
#include <vector>
#include <algorithm>
#include <float.h>
#include <memory>
#include <numeric>

template<typename T>
class SoftmaxDriver : public Driver 
{
	public:
		SoftmaxDriver() : Driver() {
		mlopenCreateTensorDescriptor(&inputTensor);
		mlopenCreateTensorDescriptor(&outputTensor);
	}

	int AddCmdLineArgs();
	int ParseCmdLineArgs(int argc, char *argv[]);
	InputFlags & GetInputFlags() { return inflags; }

	int GetandSetData();
	std::vector<int> GetInputTensorLengthsFromCmdLine();

	int AllocateBuffersAndCopy();
	
	int RunForwardGPU();
	int RunForwardCPU();
	
	int RunBackwardGPU();
	int RunBackwardCPU();
	
	int VerifyBackward();
	int VerifyForward();
	~SoftmaxDriver() {
		mlopenDestroyTensorDescriptor(outputTensor);
		mlopenDestroyTensorDescriptor(inputTensor);
	}
		
	private:
	InputFlags inflags;

	mlopenTensorDescriptor_t inputTensor;
	mlopenTensorDescriptor_t outputTensor;

	std::unique_ptr<GPUMem> in_dev;
	std::unique_ptr<GPUMem> out_dev;

	std::vector<T> in;
	std::vector<T> out;
	std::vector<T> outhost;

};

template<typename T>
int SoftmaxDriver<T>::ParseCmdLineArgs(int argc, char *argv[]) {
	inflags.Parse(argc, argv); 

	if(inflags.GetValueInt("time") == 1) {
		mlopenEnableProfiling(GetHandle(), true);
	}
	return 0; 
}

template<typename T>
int SoftmaxDriver<T>::GetandSetData() {
	std::vector<int> in_len = GetInputTensorLengthsFromCmdLine();

	SetTensor4d(inputTensor, in_len);
	SetTensor4d(outputTensor, in_len);
	
	return(0);
}

template<typename T>
int SoftmaxDriver<T>::AddCmdLineArgs() {
	inflags.AddInputFlag("forw", 'F', "0", "Run only Forward Softmax (Default=0)", "int");
	inflags.AddInputFlag("batchsize", 'n', "100", "Mini-batch size (Default=100)", "int");
	inflags.AddInputFlag("in_channels", 'c', "3", "Number of Input Channels (Default=3)", "int");
	inflags.AddInputFlag("in_h", 'H', "32", "Input Height (Default=32)", "int");
	inflags.AddInputFlag("in_w", 'W', "32", "Input Width (Default=32)", "int");
	inflags.AddInputFlag("alpha", 'A', "0.0", "Softmax shift (Default=0.0)", "double");
	inflags.AddInputFlag("beta", 'B', "0.0", "Softmax scale (Default=0.0)", "double");
	inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
	inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
	inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");

	return 0;
}

template<typename T>
std::vector<int> SoftmaxDriver<T>::GetInputTensorLengthsFromCmdLine() {
	int in_n = inflags.GetValueInt("batchsize");
	int in_c = inflags.GetValueInt("in_channels");
	int in_h = inflags.GetValueInt("in_h");
	int in_w = inflags.GetValueInt("in_w");

	return std::vector<int> ({in_n, in_c, in_h, in_w});
}

template<typename T>
int SoftmaxDriver<T>::AllocateBuffersAndCopy() {
	
	size_t in_sz = GetTensorSize(inputTensor); 
	size_t out_sz = GetTensorSize(outputTensor); 

	cl_context ctx;

	clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, NULL);

	in_dev = std::unique_ptr<GPUMem>( new GPUMem(ctx, in_sz, sizeof(float)));
	out_dev = std::unique_ptr<GPUMem> (new GPUMem(ctx, out_sz, sizeof(float)));
	

	in = std::vector<float>(in_sz);
	out = std::vector<float>(out_sz, 0);
	outhost = std::vector<float>(out_sz, 0);

	for (int i = 0; i < in_sz; i++) {
		in[i] = (T)((double)rand() * (1.0 / RAND_MAX));
	}

	cl_int status;
	status = in_dev->ToGPU(q, in.data());
	status |= out_dev->ToGPU(q, out.data());

	if(status != CL_SUCCESS) 
		printf("Error copying data to GPU\n");

	return mlopenStatusSuccess;
}

template<typename T>
int SoftmaxDriver<T>::RunForwardGPU() {

	int alpha = 1, beta = 1;

	mlopenSoftmaxForward(GetHandle(), 
			&alpha,
			inputTensor,
			in_dev->GetMem(),
			&beta,
			outputTensor,
			out_dev->GetMem());

	if(inflags.GetValueInt("time") == 1) {
		float time = 0.0;
		mlopenGetKernelTime(GetHandle(), &time);
		printf("GPU Kernel Time Forward Softmax Elapsed: %f ms\n", time);
	}

	out_dev->FromGPU(GetStream(), out.data());

	return mlopenStatusSuccess;
}

template<typename T>
int SoftmaxDriver<T>::RunForwardCPU() {
	return(0);
}

template<typename T>
int SoftmaxDriver<T>::RunBackwardGPU() {
	float alpha = 1., beta = 1.;

#if 0
	mlopenSoftmaxBackward(GetHandle(),
		&alpha,
		outputTensor,
		out_dev->GetMem(),
		dOutputTensor,
		dout_dev->GetMem(),
		inputTensor,
		in_dev->GetMem(),
		&beta,
		dInputTensor,
		din_dev->GetMem());
	
	if(inflags.GetValueInt("time") == 1) {
		float time = 0.0;
		mlopenGetKernelTime(GetHandle(), &time);
		printf("GPU Kernel Time Backward Softmax Elapsed: %f ms\n", time);
	}

	din_dev->FromGPU(GetStream(), din.data());
#endif
	return(0);

}

template<typename T>
int SoftmaxDriver<T>::VerifyForward() {
	return 0;
}

template<typename T>
int SoftmaxDriver<T>::RunBackwardCPU() {
	
	return 0;
}

template<typename T>
int SoftmaxDriver<T>::VerifyBackward() {

	return 0;
}

#endif // GUARD_MLOPEN_SOFTMAX_DRIVER_HPP
