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

#include "miopen/marginrankingloss.hpp"
#include "miopen/errors.hpp"
#include "miopen/handle.hpp"
#include "miopen/logger.hpp"
#include "miopen/tensor_ops.hpp"

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

inline void LogCmdMarginRankingLoss(bool isForward,
                                    const miopenTensorDescriptor_t targetDesc,
                                    float margin,
                                    miopenMarginRakningLossReductionMode_t reduction_mode)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(targetDesc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "marginrankinglossfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "marginrankinglossfp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "marginrankinglossbpf16";
        }

        ss << " -dims " << miopen::deref(targetDesc).GetLengths();
        ss << " -M " << margin;
        ss << " -F " << ((isForward) ? "1" : "2");
        ss << " -R " << reduction_mode;
        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t
miopenMarginRankingLossForward(miopenHandle_t handle,
                               const miopenTensorDescriptor_t input1Desc,
                               const void* input1,
                               const miopenTensorDescriptor_t input2Desc,
                               const void* input2,
                               const miopenTensorDescriptor_t targetDesc,
                               const void* target,
                               const miopenTensorDescriptor_t outputDesc,
                               void* output,
                               float margin,
                               miopenMarginRakningLossReductionMode_t reduction_mode)
{
    MIOPEN_LOG_FUNCTION(
        handle, input1Desc, input2Desc, targetDesc, outputDesc, margin, reduction_mode);
    LogCmdMarginRankingLoss(true, targetDesc, margin, reduction_mode);
    return miopen::try_([&] {
        miopen::marginrankingloss::MarginRankingLossForward(miopen::deref(handle),
                                                            miopen::deref(input1Desc),
                                                            DataCast(input1),
                                                            miopen::deref(input2Desc),
                                                            DataCast(input2),
                                                            miopen::deref(targetDesc),
                                                            DataCast(target),
                                                            miopen::deref(outputDesc),
                                                            DataCast(output),
                                                            margin,
                                                            reduction_mode);
    });
}

extern "C" miopenStatus_t
miopenMarginRankingLossBackward(miopenHandle_t handle,
                                const miopenTensorDescriptor_t input1Desc,
                                const void* input1,
                                const miopenTensorDescriptor_t input2Desc,
                                const void* input2,
                                const miopenTensorDescriptor_t targetDesc,
                                const void* target,
                                const miopenTensorDescriptor_t outGradDesc,
                                void* outGrad,
                                const miopenTensorDescriptor_t in1GradDesc,
                                void* in1Grad,
                                const miopenTensorDescriptor_t in2GradDesc,
                                void* in2Grad,
                                float margin,
                                miopenMarginRakningLossReductionMode_t reduction_mode)
{
    MIOPEN_LOG_FUNCTION(handle,
                        input1Desc,
                        input2Desc,
                        targetDesc,
                        outGradDesc,
                        in1GradDesc,
                        in2GradDesc,
                        margin,
                        reduction_mode);
    LogCmdMarginRankingLoss(false, targetDesc, margin, reduction_mode);
    return miopen::try_([&] {
        miopen::marginrankingloss::MarginRankingLossBackward(miopen::deref(handle),
                                                             miopen::deref(input1Desc),
                                                             DataCast(input1),
                                                             miopen::deref(input2Desc),
                                                             DataCast(input2),
                                                             miopen::deref(targetDesc),
                                                             DataCast(target),
                                                             miopen::deref(outGradDesc),
                                                             DataCast(outGrad),
                                                             miopen::deref(in1GradDesc),
                                                             DataCast(in1Grad),
                                                             miopen::deref(in2GradDesc),
                                                             DataCast(in2Grad),
                                                             margin,
                                                             reduction_mode);
    });
}
