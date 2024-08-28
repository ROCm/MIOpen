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

#include <miopen/handle.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/prelu.hpp>

inline std::ostream& operator<<(std::ostream& os, const std::vector<size_t>& v)
{
    os << '{';
    for(int i = 0; i < v.size(); ++i)
    {
        if(i != 0)
            os << ',';
        os << v[i];
    }
    os << '}';
    return os;
}

static void LogCmdPReLU(const miopenTensorDescriptor_t inputDesc,
                        const miopenTensorDescriptor_t weightDesc,
                        bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(inputDesc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "prelufp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "prelufp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "prelubfp16";
        }

        MIOPEN_LOG_FUNCTION(inputDesc, weightDesc);
        ss << " --input " << miopen::deref(inputDesc).GetLengths();
        ss << " --weight " << miopen::deref(weightDesc).GetLengths();
        ss << " -F " << ((is_fwd) ? "1" : "2");

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t
miopenGetPReLUBackwardWorkspaceSize(miopenHandle_t handle,
                                    const miopenTensorDescriptor_t inputDesc,
                                    const miopenTensorDescriptor_t weightDesc,
                                    size_t* sizeInBytes)
{

    MIOPEN_LOG_FUNCTION(handle, inputDesc, weightDesc, sizeInBytes);

    return miopen::try_([&] {
        miopen::deref(sizeInBytes) = miopen::GetPReLUBackwardWorkspaceSize(
            miopen::deref(handle), miopen::deref(inputDesc), miopen::deref(weightDesc));
    });
}

extern "C" miopenStatus_t miopenPReLUBackward(miopenHandle_t handle,
                                              void* workspace,
                                              const size_t workspaceSizeInBytes,
                                              const miopenTensorDescriptor_t inputDesc,
                                              const void* input,
                                              const miopenTensorDescriptor_t weightDesc,
                                              const void* weight,
                                              const miopenTensorDescriptor_t doutputDesc,
                                              const void* doutput,
                                              const miopenTensorDescriptor_t dinputDesc,
                                              void* dinput,
                                              const miopenTensorDescriptor_t dweightDesc,
                                              void* dweight)
{
    MIOPEN_LOG_FUNCTION(handle,
                        workspace,
                        workspaceSizeInBytes,
                        inputDesc,
                        input,
                        weightDesc,
                        weight,
                        doutputDesc,
                        doutput,
                        dinputDesc,
                        dinput,
                        dweightDesc,
                        dweight);

    LogCmdPReLU(inputDesc, weightDesc, false);
    return miopen::try_([&] {
        miopen::PReLUBackward(miopen::deref(handle),
                              DataCast(workspace),
                              workspaceSizeInBytes,
                              miopen::deref(inputDesc),
                              DataCast(input),
                              miopen::deref(weightDesc),
                              DataCast(weight),
                              miopen::deref(doutputDesc),
                              DataCast(doutput),
                              miopen::deref(dinputDesc),
                              DataCast(dinput),
                              miopen::deref(dweightDesc),
                              DataCast(dweight));
    });
}
