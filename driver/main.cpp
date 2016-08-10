#include <iostream>
#include <cstdio>
#include "driver.hpp"
#include "mloConvHost.hpp"
#include "mloPoolingHost.hpp"

void PrintConvParameters(std::vector<int> &in_len,
		std::vector<int> &wei_len,
		std::vector<int> &out_len) {
	printf("\nConvolution Parameters: \n");
	printf("Input Tensor Dimensions:\n");
	for(auto l : in_len)
		printf("%d, ", l);
	
	printf("\nWeight Tensor Dimensions:\n");
	for(auto l : wei_len)
		printf("%d, ", l);
	
	printf("\nOutput Tensor Dimensions:\n");
	for(auto l : out_len)
		printf("%d, ", l);
	printf("\n");
}

int main(int argc, char* argv[]) {
	ConvDriver<float> drv;
	drv.AddCmdLineArgs();
	drv.ParseCmdLineArgs(argc, argv);

	std::vector<int> in_len = drv.GetInputTensorLengthsFromCmdLine();
	std::vector<int> wei_len = drv.GetWeightTensorLengthsFromCmdLine();

	drv.SetInputTensor4d(in_len);
	drv.SetWeightTensor4d(wei_len);
	drv.SetConvDescriptorFromCmdLineArgs();

	std::vector<int> out_len = drv.GetOutputTensorLengths();
	drv.SetOutputTensor4d(out_len);

	if(drv.GetInputFlags().GetValueInt("printconv") == 1) {
		PrintConvParameters(in_len, wei_len, out_len);
	}

	drv.AllocateBuffersAndCopy();
	drv.FindForwardConvAlgo();
	drv.RunForwardConvGPU();

	if(drv.GetInputFlags().GetValueInt("verify") == 1) {
		drv.RunForwardConvCPU();
		drv.VerifyForwardConv();
	}

	// Run backward pass
	if(drv.GetInputFlags().GetValueInt("forwconv") == 0) {
		drv.FindBackwardDataAlgo();
		drv.RunBackwardDataGPU();
	}

	// pooling
	{
		mlopenHandle_t handle;
		cl_command_queue q;

		mlopenTensorDescriptor_t inputTensor;
		mlopenTensorDescriptor_t outputTensor;
		mlopenPoolingDescriptor_t poolDesc;

		std::unique_ptr<GPUMem> in_dev;
		std::unique_ptr<GPUMem> wei_dev;
		std::unique_ptr<GPUMem> out_dev;

		std::vector<float> in;
		std::vector<float> out;
		std::vector<float> outhost;


		handle = drv.GetHandle();
		mlopenGetStream(handle, &q);

		mlopenCreateTensorDescriptor(&inputTensor);
		mlopenCreateTensorDescriptor(&outputTensor);

		mlopenCreatePoolingDescriptor(&poolDesc);

		mlopenPoolingMode_t	mode = mlopenPoolingMax;
		int	windowHeight = 3;
		int	windowWidth = 3;
		int	pad_h = 1;
		int	pad_w = 1;
		int	u = 2;
		int	v = 2;
		mlopenSet2dPoolingDescriptor(poolDesc, mode, windowHeight, windowWidth, pad_h, pad_w, u, v);

		mlopenSet4dTensorDescriptor(
			inputTensor,
			mlopenFloat,
			UNPACK_VEC4(in_len));

		int out_n, out_c, out_h, out_w;
		mlopenGetPoolingForwardOutputDim(poolDesc, inputTensor, &out_n, &out_c, &out_h, &out_w);
		std::vector<int> out_pooling_len({ out_n, out_c, out_h, out_w });
		mlopenSet4dTensorDescriptor(
			outputTensor,
			mlopenFloat,
			UNPACK_VEC4(out_pooling_len));

		int n_l, c_l, h_l, w_l;
		mlopenGet4dTensorDescriptorLengths(inputTensor, &n_l, &c_l, &h_l, &w_l);

		size_t in_sz = n_l * c_l * h_l *w_l;

		mlopenGet4dTensorDescriptorLengths(outputTensor, &n_l, &c_l, &h_l, &w_l);

		size_t out_sz = n_l * c_l * h_l *w_l;

		cl_context ctx;
		clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, NULL);

		in_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(float)));
		out_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(float)));

		in = std::vector<float>(in_sz);
		out = std::vector<float>(out_sz, 0);
		outhost = std::vector<float>(out_sz, 0);

		for (int i = 0; i < in_sz; i++) {
			in[i] = rand() * (1.0 / RAND_MAX);
		}
		

		cl_int status;
		status = in_dev->ToGPU(q, in.data());
		status |= out_dev->ToGPU(q, out.data());

		if (status != CL_SUCCESS)
			printf("Error copying data to GPU\n");

		float alpha = 1., beta = 1.;
		mlopenPoolingForward(handle, poolDesc, &alpha, inputTensor, in_dev->GetMem(), &beta, outputTensor, out_dev->GetMem());

	}
	return 0;
}
