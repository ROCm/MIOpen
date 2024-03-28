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
#include "driver.hpp"
#include "m_activ.hpp"
#include "m_argmax.hpp"
#include "m_bnorm.hpp"
#include "m_cat.hpp"
#include "m_conv.hpp"
#include "m_dropout.hpp"
#include "m_fusion.hpp"
#include "m_gemm.hpp"
#include "m_groupnorm.hpp"
#include "m_layernorm.hpp"
#include "m_lrn.hpp"
#include "m_pool.hpp"
#include "m_reduce.hpp"
#include "m_rnn.hpp"
#include "m_softmax.hpp"
#include "m_sum.hpp"
#include "m_tensorop.hpp"

#include <miopen/config.h>
#include <miopen/stringutils.hpp>

#include <cstdio>
#include <iostream>

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
    if(drv == nullptr)
        drv = makeDriverConvfp16(base_arg);
    if(drv == nullptr)
        drv = makeDriverConvbfp16(base_arg);
    if(drv == nullptr)
        drv = makeDriverConvint8(base_arg);
    if(drv == nullptr)
        drv = makeDriverConvfp8(base_arg);
    if(drv == nullptr)
        drv = makeDriverConvbfp8(base_arg);
    if(drv == nullptr)
        drv = makeDriverFusion(base_arg);
    if(drv == nullptr)
        drv = makeDriverPool(base_arg);
    if(drv == nullptr)
        drv = makeDriverLrn(base_arg);
    if(drv == nullptr)
        drv = makeDriverActiv(base_arg);
    if(drv == nullptr)
        drv = makeDriverSoftmax(base_arg);
    if(drv == nullptr)
        drv = makeDriverGemm(base_arg);
    if(drv == nullptr)
        drv = makeDriverBnorm(base_arg);
    if(drv == nullptr)
        drv = makeDriverRnn(base_arg);
    if(drv == nullptr)
        drv = makeDriverDropout(base_arg);
    if(drv == nullptr)
        drv = makeDriverGroupnorm(base_arg);
    if(drv == nullptr)
        drv = makeDriverTensorop(base_arg);
    if(drv == nullptr)
        drv = makeDriverReduce(base_arg);
    if(drv == nullptr)
        drv = makeDriverLayernorm(base_arg);
    if(drv == nullptr)
        drv = makeDriverSum(base_arg);
    if(drv == nullptr)
        drv = makeDriverArgmax(base_arg);
    if(drv == nullptr)
        drv = makeDriverCat(base_arg);
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
