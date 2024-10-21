/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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
#include "miopen/miopen.h"
#include <miopen/any.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>
// #include <sstream>

// #include "miopen/common.hpp"
// #include "miopen/miopen.h"

// static void
// LogCmdAny(const miopenTensorDescriptor_t inputDesc, const int32_t dim, const bool keepdim)
// {
//     if(miopen::IsLoggingCmd())
//     {
//         std::stringstream ss;
//         auto dtype = miopen::deref(inputDesc).GetType();
//         if(dtype == miopenHalf)
//         {
//             ss << "anyfp16";
//         }
//         else if(dtype == miopenFloat)
//         {
//             ss << "anyfp32";
//         }
//         else if(dtype == miopenBFloat16)
//         {
//             ss << "anybfp16";
//         }
//         else if(dtype == miopenDouble)
//         {
//             ss << "anyfp64";
//         }

//         // int32_t size = {0};

//         // miopenGetTensorDescriptorSize(inputDesc, &size);
//         // ss << "-n " << miopen::deref(inputDesc).GetLengths()[0] << " -c " << size;

//         ss << " -dim " << dim;
//         ss << " -keepdim " << keepdim;

//         MIOPEN_LOG_DRIVER_CMD(ss.sinputtr());
//     }
// }

extern "C" miopenStatus_t miopenGetAnyWorkspaceSize(miopenHandle_t handle,
                                                    const miopenTensorDescriptor_t inputDesc,
                                                    int32_t dim,
                                                    bool keepdim,
                                                    const miopenTensorDescriptor_t outputDesc,
                                                    size_t* sizeInBytes)
{
    MIOPEN_LOG_FUNCTION(handle, inputDesc, dim, keepdim, outputDesc, sizeInBytes);

    return miopen::try_([&] {
        miopen::deref(sizeInBytes) = miopen::GetAnyWorkspaceSize(miopen::deref(handle),
                                                                 miopen::deref(inputDesc),
                                                                 miopen::deref(outputDesc),
                                                                 dim,
                                                                 keepdim);
    });
}

extern "C" miopenStatus_t miopenAnyForward(miopenHandle_t handle,
                                           void* workspace,
                                           size_t workspaceSizeInBytes,
                                           const miopenTensorDescriptor_t inputDesc,
                                           const void* input,
                                           const int32_t dim,
                                           const bool keepdim,
                                           const miopenTensorDescriptor_t outputDesc,
                                           void* output)
{
    MIOPEN_LOG_FUNCTION(handle,
                        workspace,
                        workspaceSizeInBytes,
                        inputDesc,
                        input,
                        dim,
                        keepdim,
                        outputDesc,
                        output);

    // LogCmdAny(inputDesc, dim, keepdim);

    return miopen::try_([&] {
        miopen::AnyForward(miopen::deref(handle),
                           DataCast(workspace),
                           workspaceSizeInBytes,
                           miopen::deref(inputDesc),
                           DataCast(input),
                           dim,
                           keepdim,
                           miopen::deref(outputDesc),
                           DataCast(output));
    });
}
