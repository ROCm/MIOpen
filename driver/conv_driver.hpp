#ifndef GUARD_MLOPEN_CONV_DRIVER_HPP
#define GUARD_MLOPEN_CONV_DRIVER_HPP

#include <cstdlib>
#include <mlopen.h>
#include "driver.hpp"
#include "InputFlags.hpp"
#include "tensor_driver.hpp"
#include "util_driver.hpp"
#include <mlopen/tensor.hpp>
#include <vector>
#include <algorithm>
#include <float.h>
#include <memory>
#include <numeric>
#include <../test/verify.hpp>
#include "timer.hpp"

template<typename T>
class ConvDriver : public Driver 
{
	public:
	ConvDriver() : Driver() {
		mlopenCreateTensorDescriptor(&inputTensor);
		mlopenCreateTensorDescriptor(&weightTensor);
		mlopenCreateTensorDescriptor(&outputTensor);

		mlopenCreateConvolutionDescriptor(&convDesc);
	}

	int AddCmdLineArgs();
	int ParseCmdLineArgs(int argc, char *argv[]);
	InputFlags & GetInputFlags() { return inflags; }

	int GetandSetData();
	std::vector<int> GetInputTensorLengthsFromCmdLine();
	std::vector<int> GetWeightTensorLengthsFromCmdLine();

	int SetConvDescriptorFromCmdLineArgs();

	std::vector<int> GetOutputTensorLengths();

	int AllocateBuffersAndCopy();
	
	int FindForward();
	int RunForwardGPU();
	int RunForwardCPU();
	
	int FindBackwardData();
	int FindBackwardWeights();
	int RunBackwardGPU();
	int RunBackwardDataCPU();
	int RunBackwardWeightsCPU();
	
	int VerifyBackward();
	int VerifyForward();
	~ConvDriver() {

		mlopenDestroyTensorDescriptor(outputTensor);
		mlopenDestroyTensorDescriptor(weightTensor);
		mlopenDestroyTensorDescriptor(inputTensor);

		mlopenDestroyConvolutionDescriptor(convDesc);

	}
		
	private:
	InputFlags inflags;

	mlopenTensorDescriptor_t inputTensor;
	mlopenTensorDescriptor_t weightTensor;
	mlopenTensorDescriptor_t outputTensor;

	std::unique_ptr<GPUMem> in_dev;
	std::unique_ptr<GPUMem> wei_dev;
	std::unique_ptr<GPUMem> out_dev;
	std::unique_ptr<GPUMem> workspace_dev;

	std::vector<T> in;
	std::vector<T> wei;
	std::vector<T> out;
	std::vector<T> workspace;
	std::vector<T> outhost;
	std::vector<T> inhost;
	std::vector<T> workspace_host;

	mlopenConvolutionDescriptor_t convDesc;
};

template<typename T>
int ConvDriver<T>::ParseCmdLineArgs(int argc, char *argv[]) { 
	inflags.Parse(argc, argv); 

	if(inflags.GetValueInt("time") == 1) {
		mlopenEnableProfiling(GetHandle(), true);
	}
	return 0; 
}

template<typename T>
int ConvDriver<T>::GetandSetData() {
	std::vector<int> in_len = GetInputTensorLengthsFromCmdLine();
	std::vector<int> wei_len = GetWeightTensorLengthsFromCmdLine();

	SetTensor4d(inputTensor, in_len);
	SetTensor4d(weightTensor, wei_len);
	
	SetConvDescriptorFromCmdLineArgs();

	std::vector<int> out_len = GetOutputTensorLengths();
	SetTensor4d(outputTensor, out_len);
	return(0);
}

template<typename T>
int ConvDriver<T>::AddCmdLineArgs() {
	inflags.AddInputFlag("forw", 'F', "0", "Run only Forward Convolution (Default=0)", "int");
	inflags.AddInputFlag("batchsize", 'n', "100", "Mini-batch size (Default=100)", "int");
	inflags.AddInputFlag("in_channels", 'c', "3", "Number of Input Channels (Default=3)", "int");
	inflags.AddInputFlag("in_h", 'H', "32", "Input Height (Default=32)", "int");
	inflags.AddInputFlag("in_w", 'W', "32", "Input Width (Default=32)", "int");
	inflags.AddInputFlag("out_channels", 'k', "32", "Number of Output Channels (Default=32)", "int");
	inflags.AddInputFlag("fil_h", 'y', "3", "Filter Height (Default=3)", "int");
	inflags.AddInputFlag("fil_w", 'x', "3", "Filter Width (Default=3)", "int");
	inflags.AddInputFlag("conv_stride_0", 'u', "1", "Convolution Stride Vertical (Default=1)", "int");
	inflags.AddInputFlag("conv_stride_1", 'v', "1", "Convolution Stride Horizontal (Default=1)", "int");
	inflags.AddInputFlag("pad_h", 'p', "0", "Zero Padding Height (Default=0)", "int");
	inflags.AddInputFlag("pad_w", 'q', "0", "Zero Padding Width (Default=0)", "int");
	inflags.AddInputFlag("pad_val", 'r', "0", "Padding Value (Default=0)", "int");
	inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
	inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
	inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
	inflags.AddInputFlag("wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");
	inflags.AddInputFlag("search", 's', "0", "Search Kernel Config (Default=0)", "int");
	inflags.AddInputFlag("printconv", 'P', "1", "Print Convolution Dimensions (Default=1)", "int");

	return 0;
}

template<typename T>
std::vector<int> ConvDriver<T>::GetInputTensorLengthsFromCmdLine() {
	int in_n = inflags.GetValueInt("batchsize");
	int in_c = inflags.GetValueInt("in_channels");
	int in_h = inflags.GetValueInt("in_h");
	int in_w = inflags.GetValueInt("in_w");

	return std::vector<int> ({in_n, in_c, in_h, in_w});
}

template<typename T>
std::vector<int> ConvDriver<T>::GetWeightTensorLengthsFromCmdLine() {
	int wei_n = inflags.GetValueInt("out_channels");
	int wei_c = inflags.GetValueInt("in_channels");
	int wei_h = inflags.GetValueInt("fil_h");
	int wei_w = inflags.GetValueInt("fil_w");

	return std::vector<int> ({wei_n, wei_c, wei_h, wei_w});
}

template<typename T>
int ConvDriver<T>::SetConvDescriptorFromCmdLineArgs() {

	mlopenConvolutionMode_t mode = mlopenConvolution;
	int pad_h = inflags.GetValueInt("pad_h");
	int pad_w = inflags.GetValueInt("pad_w");
	int u = inflags.GetValueInt("conv_stride_0");
	int v = inflags.GetValueInt("conv_stride_1");
	return mlopenInitConvolutionDescriptor(convDesc, mode,	pad_h, pad_w, u, v,	1, 1);
}

template<typename T>
std::vector<int> ConvDriver<T>::GetOutputTensorLengths() {
	int n, c, h, w;

	mlopenGetConvolutionForwardOutputDim(convDesc,
			inputTensor,
			weightTensor,
			&n, &c, &h, &w);

	return std::vector<int> ({n, c, h, w});
}

template<typename T>
int ConvDriver<T>::AllocateBuffersAndCopy() {
	
	size_t in_sz = GetTensorSize(inputTensor); 
	size_t wei_sz = GetTensorSize(weightTensor); 
	size_t out_sz = GetTensorSize(outputTensor); 
	size_t workSpaceSize = 0; 
	mlopenConvolutionBackwardWeightsGetWorkSpaceSize(outputTensor, weightTensor, &workSpaceSize);

	cl_context ctx;

	clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, NULL);

	in_dev = std::unique_ptr<GPUMem>( new GPUMem(ctx, in_sz, sizeof(float)));
	wei_dev = std::unique_ptr<GPUMem>( new GPUMem(ctx, wei_sz, sizeof(float)));
	out_dev = std::unique_ptr<GPUMem> (new GPUMem(ctx, out_sz, sizeof(float)));
	workspace_dev = std::unique_ptr<GPUMem> (new GPUMem(ctx, workSpaceSize/sizeof(T), sizeof(T)));
	
	in = std::vector<T>(in_sz);
	wei = std::vector<T>(wei_sz);
	out = std::vector<T>(out_sz, 0);
	workspace = std::vector<T>(workSpaceSize/sizeof(T), 0);
	outhost = std::vector<T>(out_sz, 0);
	inhost = std::vector<T>(in_sz, 0);
	workspace_host = std::vector<T>(workSpaceSize/sizeof(T), 0);

	for(int i = 0; i < in_sz; i++) {
		in[i] = (T)((double)rand() * (1.0 / RAND_MAX));
	}
	for (int i = 0; i < wei_sz; i++) {
		wei[i] = (T)((double)(rand() * (1.0 / RAND_MAX) - 0.5) * 0.001);
	}
	
	cl_int status;
	status = in_dev->ToGPU(q, in.data());
	status |= wei_dev->ToGPU(q, wei.data());
	status |= out_dev->ToGPU(q, out.data());
	status |= workspace_dev->ToGPU(q, workspace.data());
	
	if(status != CL_SUCCESS) 
		printf("Error copying data to GPU\n");

	return mlopenStatusSuccess;
}

template<typename T>
int ConvDriver<T>::FindForward() {

	int ret_algo_count;
	mlopenConvAlgoPerf_t perf;

	return mlopenFindConvolutionForwardAlgorithm(GetHandle(),
			inputTensor,
			in_dev->GetMem(),
			weightTensor,
			wei_dev->GetMem(),
			convDesc,
			outputTensor,
			out_dev->GetMem(),
			1,
			&ret_algo_count,
			&perf,
			mlopenConvolutionFastest,
			NULL,
			10,
			(inflags.GetValueInt("search")==1)?true:false
	);
}

template<typename T>
int ConvDriver<T>::RunForwardGPU() {

	FindForward();
	int alpha = 1, beta = 1;

	Timer t;
	START_TIME;

	for(int i = 0; i < inflags.GetValueInt("iter"); i++) {
	mlopenConvolutionForward(GetHandle(),
			&alpha,
			inputTensor,
			in_dev->GetMem(),
			weightTensor,
			wei_dev->GetMem(),
			convDesc,
			mlopenConvolutionFwdAlgoDirect,
			&beta,
			outputTensor,
			out_dev->GetMem(),
			NULL,
			0);
	}

	if(inflags.GetValueInt("time") == 1) {
		float time = 0.0;
		mlopenGetKernelTime(GetHandle(), &time);

		STOP_TIME;
		if(WALL_CLOCK)
			printf("Wall-clock Time Forward Conv. Elapsed: %f ms\n", t.gettime_ms() / inflags.GetValueInt("iter"));

		printf("GPU Kernel Time Forward Conv. Elapsed: %f ms\n", time);

	}

	out_dev->FromGPU(GetStream(), out.data());

	return mlopenStatusSuccess;
}

template<typename T>
int ConvDriver<T>::RunForwardCPU() {

	int in_n, in_c, in_h, in_w;
	int in_nstride, in_cstride, in_hstride, in_wstride;
	mlopenDataType_t dt;
	mlopenGet4dTensorDescriptor(inputTensor, &dt,
			&in_n, &in_c, &in_h, &in_w,
			&in_nstride, &in_cstride, &in_hstride, &in_wstride);

	int wei_n, wei_c, wei_h, wei_w;
	int wei_nstride, wei_cstride, wei_hstride, wei_wstride;
	mlopenGet4dTensorDescriptor(weightTensor, &dt,
			&wei_n, &wei_c, &wei_h, &wei_w,
			&wei_nstride, &wei_cstride, &wei_hstride, &wei_wstride);

	int out_n, out_c, out_h, out_w;
	int out_nstride, out_cstride, out_hstride, out_wstride;
	mlopenGet4dTensorDescriptor(outputTensor, &dt,
			&out_n, &out_c, &out_h, &out_w,
			&out_nstride, &out_cstride, &out_hstride, &out_wstride);

	int u, v, pad_h, pad_w, upx, upy;
	mlopenConvolutionMode_t mode;
	mlopenGetConvolutionDescriptor(convDesc, &mode, &pad_h, &pad_w, &u, &v, &upx, &upy);

	int bias = 0;

	for(int o = 0; o < out_n; o++) { // mini-batch size
		for(int w = 0; w < out_c; w++) { // out_channels (num filters)
			for(int i = 0; i < out_h; i++) { // output_height (from getforwardoutputdim())
				int in_off_h = i * v;
				for(int j = 0; j < out_w; j++) { //output_width (from getforwardoutputdim())
					float acc = 0;
					int in_off_w = j * u;
					for(int k = 0; k < in_c; k++) { // in_channels (RGB)
						for(int x = 0; x < wei_h; x++) {
							int in_x = in_off_h - pad_h + x;
							if(in_x >= 0 && in_x < in_h) {
								for(int y = 0; y < wei_w; y++) {
									int in_y = in_off_w - pad_w + y;
									if(in_y >= 0 && in_y < in_w) {
										acc +=	in[o*in_nstride + k*in_cstride + in_x*in_w + in_y] * 
											wei[w*wei_nstride + k*wei_cstride + x*wei_hstride + y];
									}
								}
							}
						}
					}
					acc = bias != 0 ? acc+bias : acc;
					outhost[o*out_nstride + w*out_cstride + i*out_hstride + j] = acc;
				}
			}
		}
	}
	return 0;
}

template<typename T>
int ConvDriver<T>::FindBackwardData() {
	int ret_algo_count;
	mlopenConvAlgoPerf_t perf;

	return mlopenFindConvolutionBackwardDataAlgorithm(GetHandle(),
			outputTensor,
			out_dev->GetMem(),
			weightTensor,
			wei_dev->GetMem(),
			convDesc,
			inputTensor,
			in_dev->GetMem(),
			1,
			&ret_algo_count,
			&perf,
			mlopenConvolutionFastest,
			NULL,
			10,
			(inflags.GetValueInt("search") == 1) ? true : false
		);
}

template<typename T>
int ConvDriver<T>::FindBackwardWeights() {
	int ret_algo_count;
	mlopenConvAlgoPerf_t perf;

	printf("before\n");
	for(int i = 0; i < 10; i++)
		printf("%f, ", wei[i]);
	mlopenFindConvolutionBackwardWeightsAlgorithm(GetHandle(),
			outputTensor,
			out_dev->GetMem(),
			inputTensor,
			in_dev->GetMem(),
			convDesc,
			weightTensor,
			wei_dev->GetMem(),
			1,
			&ret_algo_count,
			&perf,
			mlopenConvolutionFastest,
			workspace_dev->GetMem(),
			workspace_dev->GetSize(),
			(inflags.GetValueInt("search") == 1) ? true : false
		);

	wei_dev->FromGPU(GetStream(), wei.data());

	printf("after\n");
	for(int i = 0; i < 10; i++)
		printf("%f, ", wei[i]);
	return 0;
}

template<typename T>
int ConvDriver<T>::RunBackwardGPU() {

	FindBackwardData();
	FindBackwardWeights();

	int alpha = 1, beta = 1;

	int ret = 0;
	ret = mlopenConvolutionBackwardWeights(GetHandle(),
			&alpha,
			outputTensor,
			out_dev->GetMem(),
			inputTensor,
			in_dev->GetMem(),
			convDesc,
			mlopenConvolutionBwdWeightsAlgoGEMM,
			&beta,
			weightTensor,
			wei_dev->GetMem(),
			workspace_dev->GetMem(),
			workspace_dev->GetSize());

	if(inflags.GetValueInt("time") == 1) {
		float time = 0.0;
		mlopenGetKernelTime(GetHandle(), &time);
		printf("GPU Kernel Time Backward Weights Conv. Elapsed: %f ms\n", time);
	}

	workspace_dev->FromGPU(GetStream(), workspace.data());

	Timer t;
	START_TIME;

	for(int i = 0; i < inflags.GetValueInt("iter"); i++) {
	ret = mlopenConvolutionBackwardData(GetHandle(),
			&alpha,
			outputTensor,
			out_dev->GetMem(),
			weightTensor,
			wei_dev->GetMem(),
			convDesc,
			mlopenConvolutionBwdDataAlgo_0,
			&beta,
			inputTensor,
			in_dev->GetMem(),
			NULL,
			0);
	}

	if(inflags.GetValueInt("time") == 1) {
		float time = 0.0;
		mlopenGetKernelTime(GetHandle(), &time);
	
		STOP_TIME;
		if(WALL_CLOCK)
			printf("Wall-clock Time Backward Data Conv. Elapsed: %f ms\n", t.gettime_ms() / inflags.GetValueInt("iter"));
		printf("GPU Kernel Time Backward Data Conv. Elapsed: %f ms\n", time);
	}

	in_dev->FromGPU(GetStream(), in.data());

	return ret;
}

template<typename T>
int ConvDriver<T>::RunBackwardWeightsCPU() {

	int in_n, in_c, in_h, in_w;
	int in_nstride, in_cstride, in_hstride, in_wstride;
	mlopenDataType_t dt;
	mlopenGet4dTensorDescriptor(inputTensor, &dt,
			&in_n, &in_c, &in_h, &in_w,
			&in_nstride, &in_cstride, &in_hstride, &in_wstride);

	int wei_n, wei_c, wei_h, wei_w;
	int wei_nstride, wei_cstride, wei_hstride, wei_wstride;
	mlopenGet4dTensorDescriptor(weightTensor, &dt,
			&wei_n, &wei_c, &wei_h, &wei_w,
			&wei_nstride, &wei_cstride, &wei_hstride, &wei_wstride);

	int out_n, out_c, out_h, out_w;
	int out_nstride, out_cstride, out_hstride, out_wstride;
	mlopenGet4dTensorDescriptor(outputTensor, &dt,
			&out_n, &out_c, &out_h, &out_w,
			&out_nstride, &out_cstride, &out_hstride, &out_wstride);

	int u, v, pad_h, pad_w, upx, upy;
	mlopenConvolutionMode_t mode;
	mlopenGetConvolutionDescriptor(convDesc, &mode, &pad_h, &pad_w, &u, &v, &upx, &upy);

	// im2col
	size_t in_offset;
	Im2ColCPU<T>(in, in_offset, in_c, in_h, in_w, 
			wei_h, wei_w,
			out_h, out_w,
			pad_h, pad_w,
			v, u,
			workspace_host);
	
	return 0;
}

template<typename T>
int ConvDriver<T>::RunBackwardDataCPU() {

	int in_n, in_c, in_h, in_w;
	int in_nstride, in_cstride, in_hstride, in_wstride;
	mlopenDataType_t dt;
	mlopenGet4dTensorDescriptor(inputTensor, &dt,
			&in_n, &in_c, &in_h, &in_w,
			&in_nstride, &in_cstride, &in_hstride, &in_wstride);

	int wei_n, wei_c, wei_h, wei_w;
	int wei_nstride, wei_cstride, wei_hstride, wei_wstride;
	mlopenGet4dTensorDescriptor(weightTensor, &dt,
			&wei_n, &wei_c, &wei_h, &wei_w,
			&wei_nstride, &wei_cstride, &wei_hstride, &wei_wstride);

	int out_n, out_c, out_h, out_w;
	int out_nstride, out_cstride, out_hstride, out_wstride;
	mlopenGet4dTensorDescriptor(outputTensor, &dt,
			&out_n, &out_c, &out_h, &out_w,
			&out_nstride, &out_cstride, &out_hstride, &out_wstride);

	int u, v, pad_h, pad_w, upx, upy;
	mlopenConvolutionMode_t mode;
	mlopenGetConvolutionDescriptor(convDesc, &mode, &pad_h, &pad_w, &u, &v, &upx, &upy);

	for(int o = 0; o < out_n; o++) { // mini-batch size
		for(int k = 0; k < in_c; k++) { // in_channels (RGB)
			for(int w = 0; w < out_c; w++) { // out_channels (num filters)
				for(int i = 0; i < out_h; i++) { // output_height (from getforwardoutputdim())
					int in_off_h = i * v;
					for(int j = 0; j < out_w; j++) { //output_width (from getforwardoutputdim())
						int in_off_w = j * u;
						for(int x = 0; x < wei_h; x++) {
							int in_x = in_off_h - pad_h + x;
							if(in_x >= 0 && in_x < in_h) {
								for(int y = 0; y < wei_w; y++) {
									int in_y = in_off_w - pad_w + y;
									if(in_y >= 0 && in_y < in_w) {
										inhost[o*in_nstride + k*in_cstride + in_x*in_hstride + in_y] += 
											out[o*out_nstride + w*out_cstride + i*out_hstride + j] *
											wei[w*wei_nstride + k*wei_cstride + x*wei_hstride + y];
									}
								}
							}
						}
					}
				}
			}
		}
	}
	return 0;
}

template<typename T>
int ConvDriver<T>::VerifyForward() {

	RunForwardCPU();

	auto error = rms_range(outhost, out);
	const double tolerance = 1e-6;
	if (error > tolerance)
	{
		std::cout<<"Forward Convolution Failed: " << error <<"\n";
	}
	else
	{
		printf("Forward Convolution Verifies on CPU and GPU\n");
	}

	return 0;
}

template<typename T>
int ConvDriver<T>::VerifyBackward() {

	RunBackwardWeightsCPU();

	auto error_weights = rms_range(workspace_host, workspace);
	const double tolerance = 1e-9;
	if (error_weights > tolerance)
	{
		std::cout<<"Backward Convolution Weights Failed: " << error_weights <<"\n";
	}
	else
	{
		printf("Backward Convolution Weights Verifies on CPU and GPU\n");
	}

	RunBackwardDataCPU();

	auto error_data = rms_range(inhost, in);
	if (error_data > tolerance)
	{
		std::cout<<"Backward Convolution Data Failed: " << error_data <<"\n";
	}
	else
	{
		printf("Backward Convolution Data Verifies on CPU and GPU\n");
	}

	return 0;
}

#endif // GUARD_MLOPEN_CONV_DRIVER_HPP
