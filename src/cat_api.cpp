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
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

static void LogCmdCat(const std::vector<miopenTensorDescriptor_t> xDescs, bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        for(auto xDesc : xDescs)
        {
            std::stringstream ss;
            auto dtype = miopen::deref(xDesc).GetType();
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

            std::string batch_sz;
            auto dims = miopen::deref(xDesc).GetLengths();
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
                                           const std::vector<miopenTensorDescriptor_t>& xDescs,
                                           const std::vector<void*>& xs,
                                           const miopenTensorDescriptor_t& yDesc,
                                           void* y,
                                           const int32_t dim)
{
    MIOPEN_LOG_FUNCTION(handle, xDescs, xs, yDesc, y, dim);
    LogCmdCat(xDescs, true);
    return miopen::try_([&] {
        std::vector<ConstData_t> xCast;
        std::vector<miopen::TensorDescriptor> xDescsCast;
        std::transform(xDescs.begin(),
                       xDescs.end(),
                       std::back_inserter(xDescsCast),
                       [](const auto& xDesc) { return miopen::deref(xDesc); });
        std::transform(xs.begin(), xs.end(), std::back_inserter(xCast), [](const void* x) {
            return DataCast(x);
        });
        miopen::CatForward(
            miopen::deref(handle), xDescsCast, xCast, miopen::deref(yDesc), DataCast(y), dim);
    });
}
