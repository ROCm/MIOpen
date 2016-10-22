#ifndef GUARD_MLOPEN_ACTIV_DRIVER_HPP
#define GUARD_MLOPEN_ACTIV_DRIVER_HPP

#include <cstdlib>
#include <mlopen.h>
#include "driver.hpp"
#include "mloNeuronHost.hpp"
#include "InputFlags.hpp"
#include "tensor_driver.hpp"
#include <mlopen/tensor.hpp>
#include <vector>
#include <algorithm>
#include <float.h>
#include <memory>
#include <numeric>

template<typename T>
class ActivationDriver : public Driver 
{
	public:
		ActivationDriver() : Driver() {
		mlopenCreateTensorDescriptor(&inputTensor);
		mlopenCreateTensorDescriptor(&outputTensor);

		mlopenCreateActivationDescriptor(&activDesc);
			
		mlopenCreateTensorDescriptor(&dInputTensor);
		mlopenCreateTensorDescriptor(&dOutputTensor);
	}

	int AddCmdLineArgs();
	int ParseCmdLineArgs(int argc, char *argv[]);
	InputFlags & GetInputFlags() { return inflags; }

	int GetandSetData();
	std::vector<int> GetInputTensorLengthsFromCmdLine();

	int SetActivationDescriptorFromCmdLineArgs();

	int AllocateBuffersAndCopy();
	
	int RunForwardGPU();
	int RunForwardCPU();
	
	int RunBackwardGPU();
	int RunBackwardCPU();
	
	int VerifyBackward();
	int VerifyForward();
	~ActivationDriver() {

		mlopenDestroyTensorDescriptor(outputTensor);
		mlopenDestroyTensorDescriptor(inputTensor);

		mlopenDestroyActivationDescriptor(activDesc);
	}
		
	private:
	InputFlags inflags;

	mlopenTensorDescriptor_t inputTensor;
	mlopenTensorDescriptor_t outputTensor;

	std::unique_ptr<GPUMem> in_dev;
	std::unique_ptr<GPUMem> out_dev;
	std::unique_ptr<GPUMem> scale_dev;

	std::vector<T> in;
	std::vector<T> out;
	std::vector<T> outhost;

	mlopenActivationDescriptor_t activDesc;

	mlopenTensorDescriptor_t dInputTensor;
	mlopenTensorDescriptor_t dOutputTensor;	
	std::unique_ptr<GPUMem> din_dev;
	std::unique_ptr<GPUMem> dout_dev;

	std::vector<T> din;
	std::vector<T> dout;
	std::vector<T> dinhost;

};

template<typename T>
int ActivationDriver<T>::ParseCmdLineArgs(int argc, char *argv[]) {
	inflags.Parse(argc, argv); 

	if(inflags.GetValueInt("time") == 1) {
		mlopenEnableProfiling(GetHandle(), true);
	}
	return 0; 
}

template<typename T>
int ActivationDriver<T>::GetandSetData() {
	std::vector<int> in_len = GetInputTensorLengthsFromCmdLine();

	SetTensor4d(inputTensor, in_len);
	
	SetActivationDescriptorFromCmdLineArgs();

	SetTensor4d(outputTensor, in_len);
	
	SetTensor4d(dInputTensor, in_len);
	SetTensor4d(dOutputTensor, in_len);
	return(0);
}

template<typename T>
int ActivationDriver<T>::AddCmdLineArgs() {
	inflags.AddInputFlag("forw", 'F', "0", "Run only Forward LRN Normalization (Default=0)", "int");
	inflags.AddInputFlag("batchsize", 'n', "100", "Mini-batch size (Default=100)", "int");
	inflags.AddInputFlag("in_channels", 'c', "3", "Number of Input Channels (Default=3)", "int");
	inflags.AddInputFlag("in_h", 'H', "32", "Input Height (Default=32)", "int");
	inflags.AddInputFlag("in_w", 'W', "32", "Input Width (Default=32)", "int");
	inflags.AddInputFlag("mode", 'm', "3", "Activation Mode (relu,..., see spec) (Default=3(relu))", "int");
	inflags.AddInputFlag("alpha", 'A', "0.0", "Activation shift (Default=0.0)", "double");
	inflags.AddInputFlag("beta", 'B', "0.0", "Activation scale (Default=0.0)", "double");
	inflags.AddInputFlag("power", 'P', "1.0", "Activation power (Default=1.0)", "double");
	inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
	inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
	inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");

	return 0;
}

template<typename T>
std::vector<int> ActivationDriver<T>::GetInputTensorLengthsFromCmdLine() {
	int in_n = inflags.GetValueInt("batchsize");
	int in_c = inflags.GetValueInt("in_channels");
	int in_h = inflags.GetValueInt("in_h");
	int in_w = inflags.GetValueInt("in_w");

	return std::vector<int> ({in_n, in_c, in_h, in_w});
}

template<typename T>
int ActivationDriver<T>::SetActivationDescriptorFromCmdLineArgs() {

	mlopenActivationMode_t mode; 
	double Alpha = inflags.GetValueDouble("alpha");
	double Beta = inflags.GetValueDouble("beta");
	double Power = inflags.GetValueDouble("power");
	mode = (mlopenActivationMode_t)inflags.GetValueInt("mode");

	mlopenSetActivationDescriptor(activDesc,
			mode,
			Alpha,
			Beta,
			Power);
	return(0);
}

template<typename T>
int ActivationDriver<T>::AllocateBuffersAndCopy() {
	
	size_t in_sz = GetTensorSize(inputTensor); 
	size_t out_sz = GetTensorSize(outputTensor); 

	cl_context ctx;

	clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, NULL);

	in_dev = std::unique_ptr<GPUMem>( new GPUMem(ctx, in_sz, sizeof(float)));
	out_dev = std::unique_ptr<GPUMem> (new GPUMem(ctx, out_sz, sizeof(float)));
	
	din_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(float)));
	dout_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(float)));


	in = std::vector<float>(in_sz);
	out = std::vector<float>(out_sz, 0);
	outhost = std::vector<float>(out_sz, 0);

	din = std::vector<float>(in_sz);
	dout = std::vector<float>(out_sz, 0);
	dinhost = std::vector<float>(in_sz, 0);

	for (int i = 0; i < in_sz; i++) {
		in[i] = (T)((double)rand() * (1.0 / RAND_MAX));
	}

	for (int i = 0; i < out_sz; i++) {
		dout[i] = (T)((double)(rand() * (1.0 / RAND_MAX) - 0.5) * 0.001);
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
int ActivationDriver<T>::RunForwardGPU() {

	int alpha = 1, beta = 1;

	mlopenActivationForward(GetHandle(), 
			activDesc, 
			&alpha,
			inputTensor,
			in_dev->GetMem(),
			&beta,
			outputTensor,
			out_dev->GetMem(),
			false, //inflags.GetValueInt("back"),
			NULL);

	if(inflags.GetValueInt("time") == 1) {
		float time = 0.0;
		mlopenGetKernelTime(GetHandle(), &time);
		printf("GPU Kernel Time Forward Activation Elapsed: %f ms\n", time);
	}

	out_dev->FromGPU(GetStream(), out.data());

	return mlopenStatusSuccess;
}

template<typename T>
int ActivationDriver<T>::RunForwardCPU() {
	return(0);
}

template<typename T>
int ActivationDriver<T>::RunBackwardGPU() {
	float alpha = 1., beta = 1.;

	mlopenActivationBackward(GetHandle(),
		activDesc,
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
		printf("GPU Kernel Time Backward Activation Elapsed: %f ms\n", time);
	}

	din_dev->FromGPU(GetStream(), din.data());
	return(0);

}

template<typename T>
int ActivationDriver<T>::VerifyForward() {
	const double allowedEps = (1 << 2);
	mlopenActivationMode_t	v_mode;
	double	v_Alpha;
	double	v_Beta;
	double	v_Power;

	mlopenGetActivationDescriptor(activDesc,
		&v_mode,
		&v_Alpha,
		&v_Beta,
		&v_Power);

	int match = 1;
	match = mloNeuronForwardRunHostAndVerify<T>(
		v_mode,
		(T)v_Power,
		(T)v_Alpha,
		(T)v_Beta,
		in.size(),
		in.data(),
		out.data(),
		allowedEps
		);

	if(match) printf("Forward Activation Verifies on CPU and GPU\n");
	return 0;
}

template<typename T>
int ActivationDriver<T>::RunBackwardCPU() {
	
	return 0;
}

template<typename T>
int ActivationDriver<T>::VerifyBackward() {

	const double allowedEps = (1 << 2);
	mlopenActivationMode_t	v_mode;
	double	v_Alpha;
	double	v_Beta;
	double	v_Power;

	mlopenGetActivationDescriptor(activDesc,
		&v_mode,
		&v_Alpha,
		&v_Beta,
		&v_Power);

	int match = 1;
	match = mloNeuronBackwardRunHostAndVerify<T>(
		v_mode,
		(T)v_Power,
		(T)v_Alpha,
		(T)v_Beta,
		dinhost.size(),		
		in.data(),
		out.data(),
		din.data(),
		dout.data(),
		allowedEps
		);
	if(match) printf("Backward Activation Verifies on CPU and GPU\n");
	return 0;
}

#endif // GUARD_MLOPEN_ACTIV_DRIVER_HPP
