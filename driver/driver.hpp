#ifndef _MLOPEN_ConvDriver_HPP_
#define _MLOPEN_ConvDriver_HPP_
#include <cstdio>
#include <cstdlib>
#include <mlopen.h>
#include <CL/cl.h>
#include "mloConvHost.hpp"
#include "InputFlags.hpp"
#include <mlopen/tensor_extra.hpp>
#include <vector>
#include <algorithm>
#include <float.h>
#include <memory>
#include <numeric>

#define UNPACK_VEC4(v) (v[0]), (v[1]), (v[2]), (v[3])

struct GPUMem {
	GPUMem() {};
	GPUMem(cl_context &ctx, size_t psz, size_t pdata_sz) : sz(psz), data_sz(pdata_sz) {	buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, data_sz*sz, NULL, NULL); }

	int ToGPU(cl_command_queue &q, void *p) { return clEnqueueWriteBuffer(q, buf, CL_TRUE, 0, data_sz*sz, p, 0, NULL, NULL); }
	int FromGPU(cl_command_queue &q, void *p) { return clEnqueueReadBuffer(q, buf, CL_TRUE, 0, data_sz*sz, p, 0, NULL, NULL); }

	cl_mem GetMem() { return buf; }

	~GPUMem() { clReleaseMemObject(buf); }

	cl_mem buf;
	size_t sz;
	size_t data_sz;
};

template<typename T>
class ConvDriver
{
	public:
	ConvDriver(){
		mlopenCreate(&handle);
		mlopenGetStream(handle, &q);

		mlopenCreateTensorDescriptor(&inputTensor);
		mlopenCreateTensorDescriptor(&weightTensor);
		mlopenCreateTensorDescriptor(&outputTensor);

		mlopenCreateConvolutionDescriptor(&convDesc);
	};
	int AddCmdLineArgs();
	int ParseCmdLineArgs(int argc, char *argv[]) { inflags.Parse(argc, argv); return 0; }
	InputFlags & GetInputFlags() { return inflags; }
	mlopenHandle_t GetHandle() { return handle; }
	
	std::vector<int> GetInputTensorLengthsFromCmdLine();
	std::vector<int> GetWeightTensorLengthsFromCmdLine();

	std::vector<int> GetTensorStrides(mlopenTensorDescriptor_t);
	std::vector<int> GetTensorLengths(mlopenTensorDescriptor_t);

	int SetInputTensor4d(std::vector<int> &in_len);
	int SetWeightTensor4d(std::vector<int> &wei_len);
	int SetConvDescriptorFromCmdLineArgs();

	int SetOutputTensor4d(std::vector<int> &out_len);
	std::vector<int> GetOutputTensorLengths();

	size_t GetTensorSize(mlopenTensorDescriptor_t &tensor);

	int AllocateBuffersAndCopy();
	int FindForwardConvAlgo();
	int RunForwardConvGPU();
	int RunForwardConvCPU();
	int FindBackwardDataAlgo();
	int RunBackwardDataGPU();

	int VerifyForwardConv();
	~ConvDriver() {

		mlopenDestroyTensorDescriptor(outputTensor);
		mlopenDestroyTensorDescriptor(weightTensor);
		mlopenDestroyTensorDescriptor(inputTensor);

		mlopenDestroyConvolutionDescriptor(convDesc);

		mlopenDestroy(handle);
	}
#if 0
	int RunBackwardDataCPU();
	int VerifyBackwardData();
#endif
		
	private:
	InputFlags inflags;

	mlopenHandle_t handle;
	cl_command_queue q;

	mlopenTensorDescriptor_t inputTensor;
	mlopenTensorDescriptor_t weightTensor;
	mlopenTensorDescriptor_t outputTensor;

	std::unique_ptr<GPUMem> in_dev;
	std::unique_ptr<GPUMem> wei_dev;
	std::unique_ptr<GPUMem> out_dev;

	std::vector<T> in;
	std::vector<T> wei;
	std::vector<T> out;
	std::vector<T> outhost;

	mlopenConvolutionDescriptor_t convDesc;
};

template<typename T>
int ConvDriver<T>::AddCmdLineArgs() {
	inflags.AddInputFlag("forwconv", 'F', "0", "Run only Forward Convolution (Default=0)", "int");
	inflags.AddInputFlag("batchsize", 'n', "100", "Mini-batch size (Default=100)", "int");
	inflags.AddInputFlag("in_channels", 'c', "3", "Number of Input Channels (Default=3)", "int");
	inflags.AddInputFlag("in_h", 'H', "32", "Input Height (Default=32)", "int");
	inflags.AddInputFlag("in_w", 'W', "32", "Input Width (Default=32)", "int");
	inflags.AddInputFlag("out_channels", 'k', "32", "Number of Output Channels (Default=32)", "int");
	inflags.AddInputFlag("fil_h", 'x', "3", "Filter Height (Default=3)", "int");
	inflags.AddInputFlag("fil_w", 'y', "3", "Filter Width (Default=3)", "int");
	inflags.AddInputFlag("conv_stride_0", 'u', "1", "Convolution Stride Vertical (Default=1)", "int");
	inflags.AddInputFlag("conv_stride_1", 'v', "1", "Convolution Stride Horizontal (Default=1)", "int");
	inflags.AddInputFlag("pad_h", 'p', "0", "Zero Padding Height (Default=0)", "int");
	inflags.AddInputFlag("pad_w", 'q', "0", "Zero Padding Width (Default=0)", "int");
	inflags.AddInputFlag("pad_val", 'r', "0", "Padding Value (Default=0)", "int");
	inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
	inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
	inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
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
std::vector<int> ConvDriver<T>::GetTensorStrides(mlopenTensorDescriptor_t tensor) {
	int nstride;	
	int cstride;	
	int hstride;	
	int wstride;	

	 mlopenGet4dTensorDescriptorStrides(tensor, 
			 &nstride,
			 &cstride,
			 &hstride,
			 &wstride);

	 return std::vector<int> ({nstride, cstride, hstride, wstride});
}

template<typename T>
std::vector<int> ConvDriver<T>::GetTensorLengths(mlopenTensorDescriptor_t tensor) {
	int n;	
	int c;	
	int h;	
	int w;	

	 mlopenGet4dTensorDescriptorLengths(tensor, 
			 &n, &c, &h, &w);

	 return std::vector<int> ({n, c, h, w});
}

template<typename T>
int ConvDriver<T>::SetInputTensor4d(std::vector<int> &in_len){

	return mlopenSet4dTensorDescriptor(
			inputTensor,
			mlopenFloat,
			UNPACK_VEC4(in_len));
}

template<typename T>
int ConvDriver<T>::SetWeightTensor4d(std::vector<int> &wei_len){

	return mlopenSet4dTensorDescriptor(
			weightTensor,
			mlopenFloat,
			UNPACK_VEC4(wei_len));
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
int ConvDriver<T>::SetOutputTensor4d(std::vector<int> &out_len){

	return mlopenSet4dTensorDescriptor(
			outputTensor,
			mlopenFloat,
			UNPACK_VEC4(out_len));
}

template<typename T>
size_t ConvDriver<T>::GetTensorSize(mlopenTensorDescriptor_t &tensor) {

	std::vector<int> len = GetTensorLengths(tensor);
	size_t sz = std::accumulate(len.begin(), len.end(), 1, std::multiplies<int>());

	return sz;
}

template<typename T>
int ConvDriver<T>::AllocateBuffersAndCopy() {
	
	size_t in_sz = GetTensorSize(inputTensor); 
	size_t wei_sz = GetTensorSize(weightTensor); 
	size_t out_sz = GetTensorSize(outputTensor); 

	cl_context ctx;
	clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, NULL);

	in_dev = std::unique_ptr<GPUMem>( new GPUMem(ctx, in_sz, sizeof(float)));
	wei_dev = std::unique_ptr<GPUMem>( new GPUMem(ctx, wei_sz, sizeof(float)));
	out_dev = std::unique_ptr<GPUMem> (new GPUMem(ctx, out_sz, sizeof(float)));
	
	in = std::vector<T>(in_sz);
	wei = std::vector<T>(wei_sz);
	out = std::vector<T>(out_sz, 0);
	outhost = std::vector<T>(out_sz, 0);

	for(int i = 0; i < in_sz; i++) {
		in[i] = rand() * (1.0 / RAND_MAX);
	}
	for (int i = 0; i < wei_sz; i++) {
		wei[i] = (double)(rand() * (1.0 / RAND_MAX) - 0.5) * 0.001;
	}
	
	cl_int status;
	status = in_dev->ToGPU(q, in.data());
	status |= wei_dev->ToGPU(q, wei.data());
	status |= out_dev->ToGPU(q, out.data());
	
	if(status != CL_SUCCESS) 
		printf("Error copying data to GPU\n");

	return mlopenStatusSuccess;
}

template<typename T>
int ConvDriver<T>::FindForwardConvAlgo() {

	int ret_algo_count;
	mlopenConvAlgoPerf_t perf;

	return mlopenFindConvolutionForwardAlgorithm(handle,
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
			(bool)inflags.GetValueInt("search"));
}

template<typename T>
int ConvDriver<T>::RunForwardConvGPU() {

	int alpha = 1, beta = 1;

	mlopenConvolutionForward(handle,
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

	out_dev->FromGPU(q, out.data());

	return mlopenStatusSuccess;
}

template<typename T>
int ConvDriver<T>::RunForwardConvCPU() {

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
int ConvDriver<T>::FindBackwardDataAlgo() {
	int ret_algo_count;
	mlopenConvAlgoPerf_t perf;

	return mlopenFindConvolutionBackwardDataAlgorithm(handle,
			inputTensor,
			out_dev->GetMem(),
			weightTensor,
			wei_dev->GetMem(),
			convDesc,
			outputTensor,
			in_dev->GetMem(),
			1,
			&ret_algo_count,
			&perf,
			mlopenConvolutionFastest,
			NULL,
			10);
}

template<typename T>
int ConvDriver<T>::RunBackwardDataGPU() {
	int alpha = 1, beta = 1;

	return mlopenConvolutionBackwardData(handle,
			&alpha,
			inputTensor,
			out_dev->GetMem(),
			weightTensor,
			wei_dev->GetMem(),
			convDesc,
			mlopenConvolutionBwdDataAlgo_0,
			&beta,
			outputTensor,
			in_dev->GetMem(),
			NULL,
			0);
}

template<typename T>
int ConvDriver<T>::VerifyForwardConv() {

	for(int i = 0; i < out.size(); i++) {
		T diff = std::fabs(out[i] - outhost[i]);
		if(diff > std::fabs((std::max(out[i], outhost[i])) * std::numeric_limits<T>::epsilon())) {

			printf("Output Mismatch at: %d diff: %.10f gpu: %.10f cpu: %.10f \n", i, diff, out[i], outhost[i]);
			return -1;
		}
	}
	printf("Forward Convolution Verifies on CPU and GPU\n");
	return 0;
}

#endif // _MLOPEN_ConvDriver_HPP_
