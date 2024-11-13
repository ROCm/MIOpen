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
#include <miopen/sgd.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

inline void LogCmdSGD(const miopenTensorDescriptor_t& praramInDesc, bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(praramInDesc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "sgdfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "sgdfp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "sgdbf16";
        }
        std::string batch_sz;
        auto dims = miopen::deref(praramInDesc).GetLengths();
        for(auto dim : dims)
        {
            batch_sz += std::to_string(dim);
            batch_sz += ",";
        }
        ss << " -dims " << batch_sz;
        ss << " -F " << ((is_fwd) ? "1" : "2");

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenSGDForward(miopenHandle_t handle,
                                           const miopenTensorDescriptor_t paramInDesc,
                                           const void* paramIn,
                                           const miopenTensorDescriptor_t paramOutDesc,
                                           void* paramOut,
                                           const miopenTensorDescriptor_t gradDesc,
                                           const void* grad,
                                           const miopenTensorDescriptor_t momentumBufferInDesc,
                                           const void* momentumBufferIn,
                                           const miopenTensorDescriptor_t momentumBufferOutDesc,
                                           void* momentumBufferOut,
                                           const double lr,
                                           const double momentum,
                                           const double dampening,
                                           const double weightDecay,
                                           const bool nesterov,
                                           const bool momentum_initialized)
{
    MIOPEN_LOG_FUNCTION(handle,
                        paramInDesc,
                        paramOutDesc,
                        gradDesc,
                        momentumBufferInDesc,
                        momentumBufferOutDesc,
                        lr,
                        momentum,
                        dampening,
                        weightDecay,
                        (int)nesterov,
                        (int)momentum_initialized);
    LogCmdSGD(paramInDesc, true);
    return miopen::try_([&] {
        std::vector<ConstData_t> xCast;
        std::vector<miopen::TensorDescriptor*> xDescCast;
        miopen::SGD::SGDForward(miopen::deref(handle),
                                miopen::deref(paramInDesc),
                                DataCast(paramIn),
                                miopen::deref(paramOutDesc),
                                DataCast(paramOut),
                                miopen::deref(gradDesc),
                                DataCast(grad),
                                miopen::deref(momentumBufferInDesc),
                                DataCast(momentumBufferIn),
                                miopen::deref(momentumBufferOutDesc),
                                DataCast(momentumBufferOut),
                                lr,
                                momentum,
                                dampening,
                                weightDecay,
                                nesterov,
                                momentum_initialized);
    });
}
