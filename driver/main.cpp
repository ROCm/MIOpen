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

	return 0;
}
