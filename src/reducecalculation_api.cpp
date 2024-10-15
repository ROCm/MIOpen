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

#include <miopen/reducecalculation.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

static void LogCmdReduceCalculation(const miopenTensorDescriptor_t xDesc,
                                    const miopenReduceCalculationNanPropagation_t nanPropagation,
                                    const int32_t dim,
                                    const miopenReduceCalculationOp_t reduceCalculationOp,
                                    bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(xDesc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "reducecalculationfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "reducecalculationfp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "reducecalculationbfp16";
        }

        std::string input_sz;
        auto input = miopen::deref(xDesc).GetLengths();
        for(int32_t i = 0; i < input.size(); ++i)
        {
            input_sz += std::to_string(input[i]);
            if(i != input.size() - 1)
                input_sz += "x";
        }

        ss << " -F " << ((is_fwd) ? "1" : "2") << " -n " << nanPropagation;

        ss << " -R " << dim;

        ss << " -O " << reduceCalculationOp;

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t
miopenGetReduceCalculationWorkspaceSize(miopenHandle_t handle,
                                        const miopenTensorDescriptor_t xDesc,
                                        const int32_t dim,
                                        const miopenReduceCalculationOp_t reduceCalculationOp,
                                        const miopenTensorDescriptor_t reduceDesc,
                                        size_t* sizeInBytes)
{

    MIOPEN_LOG_FUNCTION(handle, xDesc, dim, reduceDesc);

    return miopen::try_([&] {
        miopen::deref(sizeInBytes) =
            miopen::GetReduceCalculationWorkspaceSize(miopen::deref(handle),
                                                      miopen::deref(xDesc),
                                                      miopen::deref(reduceDesc),
                                                      dim,
                                                      reduceCalculationOp);
    });
};

extern "C" miopenStatus_t
miopenReduceCalculationForward(miopenHandle_t handle,
                               miopenReduceCalculationNanPropagation_t nanPropagation,
                               void* workspace,
                               size_t workspaceSizeInBytes,
                               const miopenTensorDescriptor_t xDesc,
                               const void* x,
                               const int32_t dim,
                               const miopenReduceCalculationOp_t reduceCalculationOp,
                               const miopenTensorDescriptor_t reduceDesc,
                               void* y)
{
    MIOPEN_LOG_FUNCTION(
        handle, nanPropagation, workspace, workspaceSizeInBytes, xDesc, x, dim, reduceDesc, y);

    LogCmdReduceCalculation(xDesc, nanPropagation, dim, reduceCalculationOp, true);
    return miopen::try_([&] {
        miopen::ReduceCalculationForward(miopen::deref(handle),
                                         DataCast(workspace),
                                         workspaceSizeInBytes,
                                         miopen::deref(xDesc),
                                         DataCast(x),
                                         miopen::deref(reduceDesc),
                                         DataCast(y),
                                         nanPropagation,
                                         dim,
                                         reduceCalculationOp);
    });
}
