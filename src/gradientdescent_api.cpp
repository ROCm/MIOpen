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
#include <miopen/gradientdescent.hpp>
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

inline void LogCmdGradientDescent(const miopenTensorDescriptor_t& InputDesc)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(InputDesc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "gradientdescentfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "gradientdescentfp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "gradientdescentbf16";
        }
        ss << " -in_dims " << miopen::deref(InputDesc).GetLengths();
        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenGradientDescent(miopenHandle_t handle,
                                                const miopenTensorDescriptor_t varInDesc,
                                                const void* var_in,
                                                const miopenTensorDescriptor_t varOutDesc,
                                                void* var_out,
                                                const miopenTensorDescriptor_t alphaInDesc,
                                                const void* alpha_in,
                                                const miopenTensorDescriptor_t deltaInDesc,
                                                const void* delta_in)

{
    MIOPEN_LOG_FUNCTION(handle,
                        varInDesc,
                        var_in,
                        varOutDesc,
                        var_out,
                        alphaInDesc,
                        alpha_in,
                        deltaInDesc,
                        delta_in);
    LogCmdGradientDescent(varInDesc);
    return miopen::try_([&] {
        miopen::GradientDescent::GradientDescent(miopen::deref(handle),
                                                 miopen::deref(varInDesc),
                                                 DataCast(var_in),
                                                 miopen::deref(varOutDesc),
                                                 DataCast(var_out),
                                                 miopen::deref(alphaInDesc),
                                                 DataCast(alpha_in),
                                                 miopen::deref(deltaInDesc),
                                                 DataCast(delta_in));
    });
}
