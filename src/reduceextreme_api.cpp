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

#include <miopen/reduceextreme.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

static void LogCmdReduceExtreme(const miopenTensorDescriptor_t xDesc,
                                const int32_t dim,
                                const miopenReduceExtremeOp_t reduceExtremeOp,
                                bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(xDesc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "reduceextremefp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "reduceextremefp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "reduceextremebfp16";
        }

        std::string input_sz;
        auto input = miopen::deref(xDesc).GetLengths();
        for(int32_t i = 0; i < input.size(); ++i)
        {
            input_sz += std::to_string(input[i]);
            if(i != input.size() - 1)
                input_sz += "x";
        }

        ss << " -input " << input_sz;

        ss << " -F " << ((is_fwd) ? "1" : "2");

        ss << " -R " << dim;

        ss << " -O " << reduceExtremeOp;

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenReduceExtremeForward(miopenHandle_t handle,
                                                     const miopenTensorDescriptor_t xDesc,
                                                     const void* x,
                                                     const int32_t dim,
                                                     const miopenReduceExtremeOp_t reduceExtremeOp,
                                                     const miopenTensorDescriptor_t yDesc,
                                                     void* y,
                                                     const miopenTensorDescriptor_t indiceDesc,
                                                     void* indice)
{

    if((reduceExtremeOp == MIOPEN_REDUCE_EXTREME_ARGMIN) ||
       reduceExtremeOp == MIOPEN_REDUCE_EXTREME_ARGMAX)
    {
        MIOPEN_LOG_FUNCTION(handle, xDesc, x, dim, reduceExtremeOp, indiceDesc, indice);

        LogCmdReduceExtreme(xDesc, dim, reduceExtremeOp, true);

        return miopen::try_([&] {
            miopen::ReduceExtremeForward(miopen::deref(handle),
                                         miopen::deref(xDesc),
                                         DataCast(x),
                                         miopen::deref(indiceDesc),
                                         DataCast(indice),
                                         dim,
                                         reduceExtremeOp);
        });
    }
    else
    {
        MIOPEN_LOG_FUNCTION(handle, xDesc, x, dim, reduceExtremeOp, yDesc, y, indiceDesc, indice);

        LogCmdReduceExtreme(xDesc, dim, reduceExtremeOp, true);
        return miopen::try_([&] {
            miopen::ReduceExtremeForward(miopen::deref(handle),
                                         miopen::deref(xDesc),
                                         DataCast(x),
                                         miopen::deref(yDesc),
                                         DataCast(y),
                                         miopen::deref(indiceDesc),
                                         DataCast(indice),
                                         dim,
                                         reduceExtremeOp);
        });
    }
}
