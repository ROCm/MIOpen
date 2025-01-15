/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#include <miopen/miopen.h>
#include <miopen/matrix_diag.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

inline std::ostream& operator<<(std::ostream& os, const std::vector<size_t>& v)
{
    os << '{';
    for(int input = 0; input < v.size(); ++input)
    {
        if(input != 0)
            os << ',';
        os << v[input];
    }
    os << '}';
    return os;
}

static void LogCmdMatrixSetDiag(const miopenTensorDescriptor_t inputDesc,
                                const miopenTensorDescriptor_t diagDesc,
                                const miopenTensorDescriptor_t outputDesc,
                                const int64_t diagOffset0,
                                const int64_t diagOffset1,
                                const miopenMatrixDiagAlignMode_t align,
                                bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(diagDesc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "matrixsetdiagfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "matrixsetdiagfp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "matrixsetdiagbfp16";
        }

        MIOPEN_LOG_FUNCTION(diagDesc);
        ss << " -I " << miopen::deref(inputDesc).GetLengths();
        ss << " -Si " << miopen::deref(inputDesc).GetStrides();
        ss << " -D " << miopen::deref(diagDesc).GetLengths();
        ss << " -Sd " << miopen::deref(diagDesc).GetStrides();
        ss << " -O " << miopen::deref(outputDesc).GetLengths();
        ss << " -So " << miopen::deref(outputDesc).GetStrides();
        ss << " -k0 " << diagOffset0;
        ss << " -k1 " << diagOffset1;
        ss << " -al " << align;
        ss << " -F " << ((is_fwd) ? "1" : "2");

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenMatrixSetDiagForward(miopenHandle_t handle,
                                                     miopenTensorDescriptor_t inputDesc,
                                                     const void* input,
                                                     miopenTensorDescriptor_t diagDesc,
                                                     const void* diag,
                                                     miopenTensorDescriptor_t outputDesc,
                                                     void* output,
                                                     const int64_t diagOffset0,
                                                     const int64_t diagOffset1,
                                                     const miopenMatrixDiagAlignMode_t align)
{
    MIOPEN_LOG_FUNCTION(handle,
                        inputDesc,
                        input,
                        diagDesc,
                        diag,
                        outputDesc,
                        output,
                        diagOffset0,
                        diagOffset1,
                        align);
    LogCmdMatrixSetDiag(inputDesc, diagDesc, outputDesc, diagOffset0, diagOffset1, align, true);
    return miopen::try_([&] {
        miopen::MatrixSetDiagForward(miopen::deref(handle),
                                     miopen::deref(inputDesc),
                                     DataCast(input),
                                     miopen::deref(diagDesc),
                                     DataCast(diag),
                                     miopen::deref(outputDesc),
                                     DataCast(output),
                                     diagOffset0,
                                     diagOffset1,
                                     align);
    });
}
