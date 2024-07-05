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

#include <miopen/kthvalue.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor_ops.hpp>

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

static void LogCmdKthvalue(const miopenTensorDescriptor_t inputDesc, bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(inputDesc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "kthvaluefp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "kthvaluefp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "kthvaluebfp16";
        }

        MIOPEN_LOG_FUNCTION(inputDesc);
        ss << " -n " << miopen::deref(inputDesc).GetLengths()[0];
        ss << " -T " << miopen::deref(inputDesc).GetLengths();
        ss << " -Si " << miopen::deref(inputDesc).GetStrides();
        ss << " -F " << ((is_fwd) ? "1" : "2");

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenKthvalueForward(miopenHandle_t handle,
                                                miopenTensorDescriptor_t inputDesc,
                                                const void* input,
                                                miopenTensorDescriptor_t outputDesc,
                                                void* output,
                                                miopenTensorDescriptor_t indicesDesc,
                                                size_t* indices,
                                                size_t k,
                                                int32_t dim,
                                                bool keepDim)
{
    MIOPEN_LOG_FUNCTION(
        handle, inputDesc, input, outputDesc, output, indicesDesc, indices, k, dim, keepDim);

    LogCmdKthvalue(inputDesc, true);

    return miopen::try_([&] {
        miopen::KthvalueForward(miopen::deref(handle),
                                miopen::deref(inputDesc),
                                DataCast(input),
                                miopen::deref(outputDesc),
                                DataCast(output),
                                miopen::deref(indicesDesc),
                                indices,
                                k,
                                dim,
                                keepDim);
    });
}
