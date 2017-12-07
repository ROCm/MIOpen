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
#include "activ_driver.hpp"
#include "bn_driver.hpp"
#include "conv_driver.hpp"
#include "driver.hpp"
#include "gemm_driver.hpp"
#include "lrn_driver.hpp"
#include "pool_driver.hpp"
#include "softmax_driver.hpp"
#include "rnn_driver.hpp"
#include <cstdio>
#include <iostream>

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
        drv = new ConvDriver<float>();
    }
    else if(base_arg == "pool")
    {
        drv = new PoolDriver<float>();
    }
    else if(base_arg == "lrn")
    {
        drv = new LRNDriver<float>();
    }
    else if(base_arg == "activ")
    {
        drv = new ActivationDriver<float>();
    }
    else if(base_arg == "softmax")
    {
        drv = new SoftmaxDriver<float>();
    }
    else if(base_arg == "gemm")
    {
        drv = new GemmDriver<float>();
    }
    else if(base_arg == "bnorm")
    {
        drv = new BatchNormDriver<float>();
    }
    else if(base_arg == "rnn")
    {
        drv = new RNNDriver<float>();
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

    drv->RunForwardGPU();

    if(drv->GetInputFlags().GetValueInt("verify") == 1)
    {
        if(base_arg == "gemm")
        {
            printf("GEMM verification done in the GEMM library\n");
        }
        else
        {
            drv->VerifyForward();
        }
    }

    if(drv->GetInputFlags().GetValueInt("forw") == 0)
    {
        if(!(base_arg == "gemm"))
        {
            drv->RunBackwardGPU();
            if(drv->GetInputFlags().GetValueInt("verify") == 1)
            {
                drv->VerifyBackward();
            }
        }
    }

    return 0;
}
