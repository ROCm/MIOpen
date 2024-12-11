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
#include <miopen/kerasmomentum.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
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

inline void LogCmdKerasMomentum(const miopenTensorDescriptor_t& InputDesc)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(InputDesc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "kerasmomentumfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "kerasmomentumfp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "kerasmomentumbf16";
        }
        ss << " -in_dims " << miopen::deref(InputDesc).GetLengths();
        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenKerasMomentum(miopenHandle_t handle,
                                              const miopenTensorDescriptor_t varInDesc,
                                              const void* var_in,
                                              const miopenTensorDescriptor_t varOutDesc,
                                              void* var_out,
                                              const miopenTensorDescriptor_t accumInDesc,
                                              const void* accum_in,
                                              const miopenTensorDescriptor_t accumOutDesc,
                                              void* accum_out,
                                              const miopenTensorDescriptor_t lrInDesc,
                                              const void* lr_in,
                                              const miopenTensorDescriptor_t gradInDesc,
                                              const void* grad_in,
                                              const miopenTensorDescriptor_t momentumInDesc,
                                              const void* momentum_in,
                                              const bool nesterov)
{
    MIOPEN_LOG_FUNCTION(handle,
                        varInDesc,
                        var_in,
                        varOutDesc,
                        var_out,
                        accumInDesc,
                        accum_in,
                        accumOutDesc,
                        accum_out,
                        lrInDesc,
                        lr_in,
                        gradInDesc,
                        grad_in,
                        momentumInDesc,
                        momentum_in,
                        nesterov);

    LogCmdKerasMomentum(varInDesc);
    return miopen::try_([&] {
        miopen::KerasMomentum::KerasMomentum(miopen::deref(handle),
                                             miopen::deref(varInDesc),
                                             DataCast(var_in),
                                             miopen::deref(varOutDesc),
                                             DataCast(var_out),
                                             miopen::deref(accumInDesc),
                                             DataCast(accum_in),
                                             miopen::deref(accumOutDesc),
                                             DataCast(accum_out),
                                             miopen::deref(lrInDesc),
                                             DataCast(lr_in),
                                             miopen::deref(gradInDesc),
                                             DataCast(grad_in),
                                             miopen::deref(momentumInDesc),
                                             DataCast(momentum_in),
                                             nesterov);
    });
}
