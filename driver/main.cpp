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

#include "conv.hpp"
#include "fusion.hpp"
#include "pool.hpp"

#include "activ_driver.hpp"
#include "bn_driver.hpp"
#include "CBAInferFusion_driver.hpp"
#include "driver.hpp"
#include "groupnorm_driver.hpp"
#include "gemm_driver.hpp"
#include "lrn_driver.hpp"
#include "softmax_driver.hpp"
#include "rnn_driver.hpp"
#include "rnn_seq_driver.hpp"
#include "ctc_driver.hpp"
#include "dropout_driver.hpp"
#include "tensorop_driver.hpp"
#include "reduce_driver.hpp"
#include "layernorm_driver.hpp"
#include "sum_driver.hpp"
#include "argmax_driver.hpp"
#include "cat_driver.hpp"
#include <miopen/config.h>
#include <miopen/stringutils.hpp>

int main(int argc, char* argv[])
{

    std::string base_arg = ParseBaseArg(argc, argv);

    if(base_arg == "--version")
    {
        size_t major, minor, patch;
        miopenGetVersion(&major, &minor, &patch);
        std::cout << "MIOpen (version: " << major << "." << minor << "." << patch << ")"
                  << std::endl;
        exit(0); // NOLINT (concurrency-mt-unsafe)
    }

    // show command
    std::cout << "MIOpenDriver";
    for(int i = 1; i < argc; i++)
        std::cout << " " << argv[i];
    std::cout << std::endl;

    Driver* drv = makeDriverConv(base_arg);
    if(drv != nullptr)
        drv = makeDriverFusion(base_arg);
    if(drv != nullptr)
        drv = makeDriverPool(base_arg);
    if(drv != nullptr)
    {
        if(base_arg == "lrn")
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
        else if(base_arg == "gemmfp16")
        {
            drv = new GemmDriver<float16>();
        }
#endif
        else if(base_arg == "bnorm")
        {
            drv = new BatchNormDriver<float, double>();
        }
        else if(base_arg == "bnormfp16")
        {
            drv = new BatchNormDriver<float16, double, float>();
        }
        else if(base_arg == "rnn_seq")
        {
            drv = new RNNSeqDriver<float, double>();
        }
        else if(base_arg == "rnn_seqfp16")
        {
            drv = new RNNSeqDriver<float16, double>();
        }
        else if(base_arg == "rnn")
        {
            drv = new RNNDriver<float, double>();
        }
        else if(base_arg == "rnnfp16")
        {
            drv = new RNNDriver<float16, double>();
        }
        else if(base_arg == "ctc")
        {
            drv = new CTCDriver<float>();
        }
        else if(base_arg == "dropout")
        {
            drv = new DropoutDriver<float, float>();
        }
        else if(base_arg == "dropoutfp16")
        {
            drv = new DropoutDriver<float16, float>();
        }
        else if(base_arg == "groupnorm")
        {
            drv = new GroupNormDriver<float, double>();
        }
        else if(base_arg == "groupnormfp16")
        {
            drv = new GroupNormDriver<float16, double>();
        }
        else if(base_arg == "groupnormbfp16")
        {
            drv = new GroupNormDriver<bfloat16, double>();
        }
        else if(base_arg == "tensorop")
        {
            drv = new TensorOpDriver<float, float>();
        }
        else if(base_arg == "tensoropfp16")
        {
            drv = new TensorOpDriver<float16, float>();
        }
        else if(base_arg == "reduce")
        {
            drv = new ReduceDriver<float, float>();
        }
        else if(base_arg == "reducefp16")
        {
            drv = new ReduceDriver<float16, float>();
        }
        else if(base_arg == "reducefp64")
        {
            drv = new ReduceDriver<double, double>();
        }
        else if(base_arg == "layernorm")
        {
            drv = new LayerNormDriver<float, float>();
        }
        else if(base_arg == "layernormfp16")
        {
            drv = new LayerNormDriver<float16, float>();
        }
        else if(base_arg == "layernormbfp16")
        {
            drv = new LayerNormDriver<bfloat16, float>();
        }
        else if(base_arg == "sum")
        {
            drv = new SumDriver<float, float>();
        }
        else if(base_arg == "sumfp16")
        {
            drv = new SumDriver<float16, float>();
        }
        else if(base_arg == "sumbfp16")
        {
            drv = new SumDriver<bfloat16, float>();
        }
        else if(base_arg == "argmax")
        {
            drv = new ArgmaxDriver<float, float>();
        }
        else if(base_arg == "argmaxfp16")
        {
            drv = new ArgmaxDriver<float16, float>();
        }
        else if(base_arg == "argmaxbfp16")
        {
            drv = new ArgmaxDriver<bfloat16, float>();
        }
        else if(base_arg == "cat")
        {
            drv = new CatDriver<float>();
        }
        else if(base_arg == "catfp16")
        {
            drv = new CatDriver<float16>();
        }
        else if(base_arg == "catbfp16")
        {
            drv = new CatDriver<bfloat16>();
        }
    }
    if(drv == nullptr)
    {
        printf("Incorrect BaseArg\n");
        exit(0); // NOLINT (concurrency-mt-unsafe)
    }

    drv->AddCmdLineArgs();
    int rc = drv->ParseCmdLineArgs(argc, argv);
    if(rc != 0)
    {
        std::cout << "ParseCmdLineArgs() FAILED, rc = " << rc << std::endl;
        return rc;
    }
    drv->GetandSetData();
    rc = drv->AllocateBuffersAndCopy();
    if(rc != 0)
    {
        std::cout << "AllocateBuffersAndCopy() FAILED, rc = " << rc << std::endl;
        return rc;
    }

    int fargval =
        !miopen::StartsWith(base_arg, "CBAInfer") ? drv->GetInputFlags().GetValueInt("forw") : 1;
    bool bnFwdInVer   = (fargval == 2 && miopen::StartsWith(base_arg, "bnorm"));
    bool verifyarg    = (drv->GetInputFlags().GetValueInt("verify") == 1);
    int cumulative_rc = 0; // Do not stop running tests in case of errors.

    if(fargval & 1 || fargval == 0 || bnFwdInVer)
    {
        rc = drv->RunForwardGPU();
        cumulative_rc |= rc;
        if(rc != 0)
            std::cout << "RunForwardGPU() FAILED, rc = "
                      << "0x" << std::hex << rc << std::dec << std::endl;
        if(verifyarg) // Verify even if Run() failed.
            cumulative_rc |= drv->VerifyForward();
    }

    if(fargval != 1)
    {
        rc = drv->RunBackwardGPU();
        cumulative_rc |= rc;
        if(rc != 0)
            std::cout << "RunBackwardGPU() FAILED, rc = "
                      << "0x" << std::hex << rc << std::dec << std::endl;
        if(verifyarg) // Verify even if Run() failed.
            cumulative_rc |= drv->VerifyBackward();
    }

    return cumulative_rc;
}
