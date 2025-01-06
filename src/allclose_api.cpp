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

#include <miopen/allclose.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

inline std::ostream& operator<<(std::ostream& os, const std::vector<uint64_t>& v)
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

inline void LogCmdAllClose(const miopenTensorDescriptor_t iDesc, const bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(iDesc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "allclosefp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "allclosefp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "allclosebfp16";
        }
        ss << " -Is ";
        ss << miopen::deref(iDesc).GetLengths();
        ss << " -F " << ((is_fwd) ? "1" : "2");
        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t
miopenGetAllCloseForwardWorkspaceSize(miopenHandle_t handle,
                                      const miopenTensorDescriptor_t input1Desc,
                                      const miopenTensorDescriptor_t input2Desc,
                                      const miopenTensorDescriptor_t outputDesc,
                                      size_t* sizeInBytes)
{
    MIOPEN_LOG_FUNCTION(handle, input1Desc, input2Desc);
    return miopen::try_([&] {
        miopen::deref(sizeInBytes) =
            miopen::allclose::GetAllCloseForwardWorkspaceSize(miopen::deref(handle),
                                                              miopen::deref(input1Desc),
                                                              miopen::deref(input2Desc),
                                                              miopen::deref(outputDesc));
    });
}

extern "C" miopenStatus_t miopenAllCloseForward(miopenHandle_t handle,
                                                const miopenTensorDescriptor_t input1Desc,
                                                const void* input1,
                                                const miopenTensorDescriptor_t input2Desc,
                                                const void* input2,
                                                const miopenTensorDescriptor_t outputDesc,
                                                void* output,
                                                const float atol,
                                                const float rtol,
                                                const bool equal_nan,
                                                void* workspace,
                                                const size_t workspaceSizeInBytes)
{
    MIOPEN_LOG_FUNCTION(handle,
                        input1Desc,
                        input1,
                        input2Desc,
                        input2,
                        outputDesc,
                        output,
                        atol,
                        rtol,
                        equal_nan,
                        workspace,
                        workspaceSizeInBytes);
    LogCmdAllClose(input1Desc, true);

    return miopen::try_([&] {
        miopen::allclose::AllCloseForward(miopen::deref(handle),
                                          miopen::deref(input1Desc),
                                          DataCast(input1),
                                          miopen::deref(input2Desc),
                                          DataCast(input2),
                                          miopen::deref(outputDesc),
                                          DataCast(output),
                                          atol,
                                          rtol,
                                          equal_nan,
                                          DataCast(workspace),
                                          workspaceSizeInBytes);
    });
}
