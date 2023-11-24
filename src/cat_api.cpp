/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include <miopen/cat.hpp>
#ifdef MIOPEN_BETA_API
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

static void LogCmdCat(const std::vector<miopenTensorDescriptor_t> inputDescs, bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        for(auto inputDesc : inputDescs)
        {
            std::stringstream ss;
            auto dtype = miopen::deref(inputDesc).GetType();
            if(dtype == miopenHalf)
            {
                ss << "catfp16";
            }
            else if(dtype == miopenFloat)
            {
                ss << "catfp32";
            }
            else if(dtype == miopenBFloat16)
            {
                ss << "catf16";
            }
            else if(dtype == miopenDouble)
            {
                ss << "catfp64";
            }

            std::string batch_sz;
            auto dims = miopen::deref(inputDesc).GetLengths();
            for(auto dim : dims)
            {
                batch_sz += std::to_string(dim);
                batch_sz += ",";
            }
            batch_sz.pop_back();
            ss << " -dims " << batch_sz;
            ss << " -F " << ((is_fwd) ? "1" : "2");

            MIOPEN_LOG_DRIVER_CMD(ss.str());
        }
    }
}

extern "C" miopenStatus_t miopenCatForward(miopenHandle_t handle,
                                           const std::vector<miopenTensorDescriptor_t>& inputDescs,
                                           const std::vector<void*>& inputs,
                                           const miopenTensorDescriptor_t& outputDesc,
                                           void* output,
                                           const int32_t dim)
{
    MIOPEN_LOG_FUNCTION(handle, inputDescs, inputs, outputDesc, output, dim);
    LogCmdCat(inputDescs, true);
    std::vector<ConstData_t> inputsCast;
    std::vector<miopen::TensorDescriptor> inputDescsCast;
    return miopen::try_([&] {
        //        miopen::c_array_view<const miopenTensorDescriptor_t> xDescArray{xDesc,
        //        size_t(sequenceLen)};
        std::transform(inputDescs.begin(),
                       inputDescs.end(),
                       inputDescsCast.begin(),
                       [](auto input) { return miopen::deref(input); });
        std::transform(inputs.begin(), inputs.end(), inputsCast.begin(), [](auto input) {
            return DataCast(input);
        });
        miopen::CatForward(miopen::deref(handle),
                           inputDescsCast,
                           inputsCast,
                           miopen::deref(outputDesc),
                           DataCast(output),
                           dim);
    });
}
#endif
