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
#include <miopen/gemm.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor.hpp>

#include <array>
#include <initializer_list>

extern "C" miopenStatus_t miopenCreateGemmDescriptor(miopenGemmDescriptor_t* gemmDesc)
{

    //MIOPEN_LOG_FUNCTION(gemmDesc);
    return miopen::try_([&] { miopen::deref(gemmDesc) = new miopen::GemmNewDescriptor(); });
}

static void LogCmdGemm(const miopenTensorDescriptor_t ADesc,
                       const miopenTensorDescriptor_t BDesc,
                       const miopenGemmDescriptor_t gemmDesc)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        if(miopen::deref(ADesc).GetType() == miopenHalf)
        {
            ss << "gemmfp16";
        }
        else
        {
            ss << "gemm";
        }
        ss << " -n " << miopen::deref(ADesc).GetLengths()[0]
           << " -c " << miopen::deref(ADesc).GetLengths()[1]
           << " -M " << miopen::deref(ADesc).GetLengths()[2]
           << " -K " << miopen::deref(ADesc).GetLengths()[3]
           << " -N " << miopen::deref(BDesc).GetLengths()[3]
           << " -alpha "<< miopen::deref(gemmDesc).GetAlpha()
           << " -beta " << miopen::deref(gemmDesc).GetBeta();
        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenGemm(miopenHandle_t handle,
                                     const miopenGemmDescriptor_t gemmDesc,
                                     const void* alpha,
                                     const miopenTensorDescriptor_t ADesc,
                                     const void* A,
                                     const void* beta,
                                     const miopenTensorDescriptor_t BDesc,
                                     const void* B,                                                     
                                     const miopenTensorDescriptor_t CDesc,
                                     void* C)
{

    //MIOPEN_LOG_FUNCTION(handle, gemmDesc, alpha, ADesc, A, beta, BDesc, B, CDesc, C);

    // bfloat16 not supported for activation operation
    if(miopen::deref(CDesc).GetType() == miopenBFloat16 ||
       miopen::deref(ADesc).GetType() == miopenBFloat16 ||
       miopen::deref(BDesc).GetType() == miopenBFloat16)
    {
        return miopenStatusNotImplemented;
    }
    LogCmdGemm(ADesc, BDesc, gemmDesc);
    return miopen::try_([&] {
        miopen::deref(gemmDesc).CallGemm(miopen::deref(handle),
                                         alpha,
                                         miopen::deref(ADesc),
                                         DataCast(A),
                                         beta,
                                         miopen::deref(BDesc),
                                         DataCast(B),                                        
                                         miopen::deref(CDesc),
                                         DataCast(C));
    });
}

extern "C" miopenStatus_t miopenDestroyGemmDescriptor(miopenGemmDescriptor_t gemmDesc)
{

    //MIOPEN_LOG_FUNCTION(gemmDesc);
    return miopen::try_([&] { miopen_destroy_object(gemmDesc); });
}
