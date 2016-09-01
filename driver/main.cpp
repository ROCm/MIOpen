#include <iostream>
#include <cstdio>
#include "driver.hpp"
#include "conv_driver.hpp"
#include "pool_driver.hpp"
#include "lrn_driver.hpp"
#include "activ_driver.hpp"

int main(int argc, char *argv[]) {

	std::string base_arg = ParseBaseArg(argc, argv);

	Driver *drv;
	if(base_arg == "conv") {
		drv = new ConvDriver<float>();
	}
	else if(base_arg == "pool") {
		drv = new PoolDriver<float>();
	}
	else if(base_arg == "lrn") {
		drv = new LRNDriver<float>();
	}
	else if (base_arg == "activ") {
		drv = new ActivationDriver<float>();
	}
	else {
		printf("Incorrect BaseArg\n");
		exit(0);
	}

	drv->AddCmdLineArgs();
	drv->ParseCmdLineArgs(argc, argv);
	drv->GetandSetData();

	drv->AllocateBuffersAndCopy();

	drv->RunForwardGPU();

	if(drv->GetInputFlags().GetValueInt("verify") == 1) {
		drv->VerifyForward();
	}
	
	if(drv->GetInputFlags().GetValueInt("forw") == 0) {
		drv->RunBackwardGPU();
		if(drv->GetInputFlags().GetValueInt("verify") == 1) {
			drv->VerifyBackward();
		}
	}

	return 0;
}
