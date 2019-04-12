/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <iostream>
#include <cstdio>

#include "activ_driver.hpp"
#include "bn_driver.hpp"
#include "conv_driver.hpp"
#include "CBAInferFusion_driver.hpp"
#include "driver.hpp"
#include "gemm_driver.hpp"
#include "lrn_driver.hpp"
#include "pool_driver.hpp"
#include "softmax_driver.hpp"
#include "rnn_driver.hpp"
#include "miopen/config.h"

int main(int argc, char* argv[])
{
    // show command
    std::cout << "MIOpenDriver:";
    for(int i = 1; i < argc; i++)
        std::cout << " " << argv[i];
    std::cout << std::endl;

    std::string base_arg = ParseBaseArg(argc, argv);

    Driver* drv;
    if(base_arg == "conv")
    {
        drv = new ConvDriver<float, float>();
    }
    else if(base_arg == "convfp16")
    {
        drv = new ConvDriver<float16, float>();
    }
    else if(base_arg == "convint8")
    {
        drv = new ConvDriver<int8_t, float>();
    }
    else if(base_arg == "CBAInfer")
    {
        drv = new CBAInferFusionDriver<float, double>();
    }
    else if(base_arg == "CBAInferfp16")
    {
        drv = new CBAInferFusionDriver<float16, double>();
    }
    else if(base_arg == "pool")
    {
        drv = new PoolDriver<float, double>();
    }
    else if(base_arg == "poolfp16")
    {
        drv = new PoolDriver<float16, double>();
    }
    else if(base_arg == "lrn")
    {
        drv = new LRNDriver<float, double>();
    }
    else if(base_arg == "lrnfp16")
    {
        drv = new LRNDriver<float16, double>();
    }
    else if(base_arg == "activ")
    {
        drv = new ActivationDriver<float, double>();
    }
    else if(base_arg == "activfp16")
    {
        drv = new ActivationDriver<float16, double>();
    }
    else if(base_arg == "softmax")
    {
        drv = new SoftmaxDriver<float, double>();
    }
    else if(base_arg == "softmaxfp16")
    {
        drv = new SoftmaxDriver<float16, double>();
    }
#if MIOPEN_USE_GEMM
    else if(base_arg == "gemm")
    {
        drv = new GemmDriver<float>();
    }
// TODO half is not supported in gemm
//    else if(base_arg == "gemmfp16")
//    {
//        drv = new GemmDriver<float16>();
//    }
#endif
    else if(base_arg == "bnorm")
    {
        drv = new BatchNormDriver<float, double>();
    }
    else if(base_arg == "bnormfp16")
    {
        drv = new BatchNormDriver<float16, double, float>();
    }
    else if(base_arg == "rnn")
    {
        drv = new RNNDriver<float, double>();
    }
    else if(base_arg == "rnnfp16")
    {
        drv = new RNNDriver<float16, double>();
    }
    else
    {
        printf("Incorrect BaseArg\n");
        exit(0);
    }

    drv->AddCmdLineArgs();
    drv->ParseCmdLineArgs(argc, argv);
    drv->GetandSetData();
    drv->AllocateBuffersAndCopy();

    int fargval = ((base_arg != "CBAInfer") && (base_arg != "CBAInferfp16"))
                      ? drv->GetInputFlags().GetValueInt("forw")
                      : 1;
    bool bnFwdInVer = (fargval == 2 && (base_arg == "bnorm"));
    bool verifyarg  = (drv->GetInputFlags().GetValueInt("verify") == 1);

    if(fargval & 1 || fargval == 0 || bnFwdInVer)
    {
        drv->RunForwardGPU();
        if(verifyarg)
            drv->VerifyForward();
    }

    if(fargval != 1)
    {
        drv->RunBackwardGPU();
        if(verifyarg)
        {
            drv->VerifyBackward();
        }
    }

    return 0;
}
