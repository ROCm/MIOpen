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

#ifndef NDEBUG
static inline void dumpDataToFile(const char * name, void * ptr, size_t size, const char * fileName)
{
	FILE * fp = fopen(fileName, "wb");
	if (!fp) printf("ERROR: dumpDataToFile: unable to create: %s\n");
	else {
		fwrite(ptr, 1, size, fp);
		fclose(fp);
		printf("OK: dumpDataToFile: dumped %s into %s (%d bytes)\n", name, fileName, (int)size);
	}
}
#endif

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
	std::unique_ptr<GPUMem> din_dev;
	std::unique_ptr<GPUMem> wei_dev;
	std::unique_ptr<GPUMem> dwei_dev;
	std::unique_ptr<GPUMem> out_dev;
	std::unique_ptr<GPUMem> dout_dev;
	std::unique_ptr<GPUMem> workspace_dev;

	std::vector<T> in;
	std::vector<T> din;
	std::vector<T> wei;
	std::vector<T> dwei;
	std::vector<T> out;
	std::vector<T> dout;
	std::vector<T> workspace;
	std::vector<T> outhost;
	std::vector<T> inhost;
	std::vector<T> workspace_host;
	std::vector<T> din_host;
	std::vector<T> dwei_host;

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
	mlopenConvolutionBackwardWeightsGetWorkSpaceSize(outputTensor, inputTensor, convDesc, weightTensor, &workSpaceSize);

	cl_context ctx;

	clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, NULL);

	in_dev = std::unique_ptr<GPUMem>( new GPUMem(ctx, in_sz, sizeof(float)));
	din_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(float)));
	wei_dev = std::unique_ptr<GPUMem>( new GPUMem(ctx, wei_sz, sizeof(float)));
	dwei_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, wei_sz, sizeof(float)));
	dout_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(float)));
	out_dev = std::unique_ptr<GPUMem> (new GPUMem(ctx, out_sz, sizeof(float)));
	workspace_dev = std::unique_ptr<GPUMem> (new GPUMem(ctx, workSpaceSize/sizeof(T), sizeof(T)));
	
	in = std::vector<T>(in_sz);
	din = std::vector<T>(in_sz);
	wei = std::vector<T>(wei_sz);
	dwei = std::vector<T>(wei_sz, 0);
	dout = std::vector<T>(out_sz, 0);
	out = std::vector<T>(out_sz, 0);
	workspace = std::vector<T>(workSpaceSize/sizeof(T), 0);
	outhost = std::vector<T>(out_sz, 0);
	inhost = std::vector<T>(in_sz, 0);
	workspace_host = std::vector<T>(workSpaceSize/sizeof(T), 0);
	dwei_host = std::vector<T>(wei_sz, 0);
	din_host = std::vector<T>(in_sz, 0);

	for(int i = 0; i < in_sz; i++) {
		in[i] = (T)((double)rand() * (1.0 / RAND_MAX));
	}
	for (int i = 0; i < out_sz; i++) {
		out[i] = (T)((double)rand() * (1.0 / RAND_MAX));
		dout[i] = (T)((double)rand() * (1.0 / RAND_MAX));
	}


	for (int i = 0; i < wei_sz; i++) {
		wei[i] = (T)((double)(rand() * (1.0 / RAND_MAX) - 0.5) );
	}
	
	cl_int status;
	status = in_dev->ToGPU(q, in.data());
	status |= din_dev->ToGPU(q, in.data());
	status |= wei_dev->ToGPU(q, wei.data());
	status |= dwei_dev->ToGPU(q, dwei.data());
	status |= dout_dev->ToGPU(q, dout.data());
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
			dout_dev->GetMem(),
			weightTensor,
			wei_dev->GetMem(),
			convDesc,
			inputTensor,
			din_dev->GetMem(),
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

	return mlopenFindConvolutionBackwardWeightsAlgorithm(GetHandle(),
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
}

template<typename T>
int ConvDriver<T>::RunBackwardGPU() {

	FindBackwardData();

	int alpha = 1, beta = 1;

	int ret = 0;

	Timer t;
	START_TIME;

	for(int i = 0; i < inflags.GetValueInt("iter"); i++) {
	ret = mlopenConvolutionBackwardData(GetHandle(),
			&alpha,
			outputTensor,
			dout_dev->GetMem(),
			weightTensor,
			wei_dev->GetMem(),
			convDesc,
			mlopenConvolutionBwdDataAlgo_0,
			&beta,
			inputTensor,
			din_dev->GetMem(),
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

	din_dev->FromGPU(GetStream(), din.data());


	FindBackwardWeights();
	ret = mlopenConvolutionBackwardWeights(GetHandle(),
		&alpha,
		outputTensor,
		dout_dev->GetMem(),
		inputTensor,
		in_dev->GetMem(),
		convDesc,
		mlopenConvolutionBwdWeightsAlgoDirect,
		&beta,
		weightTensor,
		dwei_dev->GetMem(),
		workspace_dev->GetMem(),
		workspace_dev->GetSize());

	if (inflags.GetValueInt("time") == 1) {
		float time = 0.0;
		mlopenGetKernelTime(GetHandle(), &time);
		printf("GPU Kernel Time Backward Weights Conv. Elapsed: %f ms\n", time);
	}
	workspace_dev->FromGPU(GetStream(), workspace.data());
	dwei_dev->FromGPU(GetStream(), dwei.data());


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

	assert(in_wstride == 1);
	assert(wei_wstride == 1);
	assert(out_wstride == 1);


	for (int o = 0; o < out_n; o++) // mini-batch size
	{
		for (int w = 0; w < out_c; w++) // out_channels (num filters)
		{
			for (int k = 0; k < in_c; k++) // in_channels (RGB)
			{
				for (int x = 0; x < wei_h; x++) // filter height
				{
					for (int y = 0; y < wei_w; y++) // filter width
					{
						for (int i = 0; i < out_h; i++) // output height
						{
							for (int j = 0; j < out_w; j++) // output width
							{
								int in_i = x + i * u - pad_h;
								int in_j = y + j * v - pad_w;

								if ((in_i >= 0) && (in_i < in_h) && (in_j >= 0) && (in_j < in_w))
								{
									dwei_host[w*wei_nstride + k*wei_cstride + x*wei_hstride + y] +=
										in[o*in_nstride + k*in_cstride + in_i*in_hstride + in_j] *
										dout[o*out_nstride + w*out_cstride + i*out_hstride + j];
								}
							}
						}
					}
				}
			}
		}
	}

#ifdef BACKWARD_WRW_VERIFY_DIRECT_2

	{
		assert(u == 1);
		assert(v == 1);

		std::fill(dwei_host.begin(), dwei_host.end(), (T)0);

		int pad0 = pad_w;
		int pad1 = pad_h;
		int batch_sz = out_n;
		int outputs = out_c;
		int inputs = in_c;
		int top_df_batch_stride = out_nstride;
		int top_df_channel_stride = out_cstride;
		int top_df_stride = out_hstride;
		int bot_batch_stride = in_nstride;
		int bot_channel_stride = in_cstride;
		int weights_df_v2_stride = wei_nstride;
		int bot_stride = in_hstride;

		int kernel_size0 = wei_w;
		int kernel_size1 = wei_h;
		int kernel_sz = kernel_size0 * kernel_size1;

		int top_height = out_h;
		int top_width = out_w;
		int bot_height = in_h;
		int bot_width = in_w;


		for (int b = 0; b < batch_sz; ++b)
		{
			for (int o = 0; o < outputs; ++o)
			{
				for (int c = 0; c < inputs; ++c)
				{
					int top_df_off = b*top_df_batch_stride + o*top_df_channel_stride;
					int bot_off = b*bot_batch_stride + c*bot_channel_stride;
					int we_off = o * weights_df_v2_stride + c * kernel_sz;

					for (int j = 0, c_j = j - pad1; j < top_height; ++j, ++c_j)
					{

						for (int i = 0, c_i = i - pad0; i < top_width; i++, ++c_i)
						{
							float top_val = dout[top_df_off + j*top_df_stride + i];

							for (int k = 0, c_j = j - pad1; k < kernel_size1; ++k, ++c_j)
							{

								for (int l = 0, c_i = i - pad0; l < kernel_size0; ++l, ++c_i)
								{

									float bot_val = (c_j >= 0 && c_j < bot_height && c_i >= 0 && c_i < bot_width) ?
										in[bot_off + c_j*bot_stride + c_i] : 0;


									dwei_host[we_off + k*kernel_size0 + l] += bot_val*top_val;


								}
							}
						}
					}
				}

			}
		}
	}

#endif

#ifdef BACKWARD_WRW_VERIFY_GEMM

	{
		assert(u == v);
		assert(pad_h == pad_w);

		std::fill(dwei_host.begin(), dwei_host.end(), (T)0);

		int batch_sz = out_n;
		int outputs = out_c;
		int inputs = in_c;

		int bot_batch_stride = in_c*in_h*in_w;
		int bot_channel_stride = in_h*in_w;
		int bot_stride = in_w;
		int bot_height = in_h;
		int bot_width = in_w;

		int top_width = out_w;
		int top_height = out_h;
		int top_df_channel_stride = top_width * top_height;
		int top_df_batch_stride = top_df_channel_stride * out_c;

		int weights_width = wei_w * wei_h * wei_c;
		int weights_height = wei_n;
		int weights_df_v_stride = weights_width;
		int kernel_size = wei_w;

		int pad = pad_w;
		int stride = v;

		// allocate raw data for in, dout, dwei for using im2col/gemm aDNN functions
		T * weights_df_v_ptr = new T[weights_width * weights_height];
		T * top_df_ptr = new T[out_n*out_c*out_h*out_w];
		T * bot_ptr = new T[in_n*in_c*in_h*in_w];

		// copy input (in) into packed
		for (int n = 0; n < in_n; n++)
		{
			for (int c = 0; c < in_c; c++)
			{
				for (int h = 0; h < in_h; h++)
				{
					for (int w = 0; w < in_w; w++)
					{
						bot_ptr[n*in_c*in_h*in_w + c*in_h*in_w + h*in_w + w] = in[n*in_nstride + c*in_cstride + h*in_hstride + w];
					}
				}
			}
		}

		// copy delta out (dout) into packed
		for (int n = 0; n < out_n; n++)
		{
			for (int c = 0; c < out_c; c++)
			{
				for (int h = 0; h < out_h; h++)
				{
					for (int w = 0; w < out_w; w++)
					{
						top_df_ptr[n*out_c*out_h*out_w + c*out_h*out_w + h*out_w + w] = dout[n*out_nstride + c*out_cstride + h*out_hstride + w];
					}
				}
			}
		}

		int im2col_batch_stride = weights_width * top_width * top_height;
		T * im2col_ptr = new T[im2col_batch_stride * batch_sz];

		#define ADNN_MM_TRANSPOSE 1
		memset(weights_df_v_ptr, 0, weights_width * weights_height * sizeof(T));
		for (int b = 0; b < batch_sz; ++b)
		{
			ADNN_im2col_cpu<T>((const T*)&bot_ptr[bot_batch_stride * b], inputs,
				bot_height, bot_width, kernel_size, pad,
				stride, &im2col_ptr[im2col_batch_stride * b]);
			// sum up over mini-batch
			ADNN_mm_cpu<T>((const T*)&top_df_ptr[top_df_batch_stride * b], top_width * top_height, outputs, top_df_channel_stride, 0,
				(const T *)&im2col_ptr[im2col_batch_stride * b], top_width * top_height, weights_width, top_width * top_height, ADNN_MM_TRANSPOSE,
				weights_df_v_ptr, weights_width, weights_height, weights_df_v_stride, 0,
				1, 1);

		}

		// read back packed delta weight
		for (int n = 0; n < wei_n; n++)
		{
			for (int c = 0; c < wei_c; c++)
			{
				for (int h = 0; h < wei_h; h++)
				{
					for (int w = 0; w < wei_w; w++)
					{
						dwei_host[n*wei_nstride + c*wei_cstride + h*wei_hstride + w] = weights_df_v_ptr[n*wei_c*wei_h*wei_w + c*wei_h*wei_w + h*wei_w + w];
					}
				}
			}
		}

		delete[] im2col_ptr;
		delete[] weights_df_v_ptr;
		delete[] top_df_ptr;
		delete[] bot_ptr;

	}

#endif

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
										din_host[o*in_nstride + k*in_cstride + in_x*in_hstride + in_y] += 
											dout[o*out_nstride + w*out_cstride + i*out_hstride + j] *
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

#ifndef NDEBUG
	//dumpDataToFile("in", in.data(), sizeof(float) * in.size(), "dump_fwd_in.bin");
	//dumpDataToFile("wei", wei.data(), sizeof(float) * wei.size(), "dump_fwd_wei.bin");
	//dumpDataToFile("out", out.data(), sizeof(float) * out.size(), "dump_fwd_out_gpu.bin");
	//dumpDataToFile("outhost", outhost.data(), sizeof(float) * outhost.size(), "dump_fwd_out_cpu.bin");
#endif

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
	const double tolerance = 1e-6;

	RunBackwardWeightsCPU();

	auto error_weights = rms_range(dwei_host, dwei);
	if (error_weights > tolerance)
	{
		std::cout<<"Backward Convolution Weights Failed: " << error_weights <<"\n";
	}
	else
	{
		printf("Backward Convolution Weights Verifies on CPU and GPU\n");
	}

	RunBackwardDataCPU();


	auto error_data = rms_range(din_host, din);

#ifndef NDEBUG
	//dumpDataToFile("dout", dout.data(), sizeof(float) * dout.size(), "dump_bwd_out.bin");
	//dumpDataToFile("wei", wei.data(), sizeof(float) * wei.size(), "dump_bwd_wei.bin");
	//dumpDataToFile("din", din.data(), sizeof(float) * din.size(), "dump_bwd_in_gpu.bin");
	//dumpDataToFile("dinhost", din_host.data(), sizeof(float) * din_host.size(), "dump_bwd_in_cpu.bin");
#endif

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
