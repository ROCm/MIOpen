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
#include <miopen/unsortedsegmentsum.hpp>
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

inline void LogCmdUnsortedSegmentSum(const miopenTensorDescriptor_t& InputDesc,
                                     const miopenTensorDescriptor_t& OutputDesc,
                                     const bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(InputDesc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "unsortedsegmentsumfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "unsortedsegmentsumfp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "unsortedsegmentsumbf16";
        }
        ss << " -in_dims " << miopen::deref(InputDesc).GetLengths();
        ss << " -out_dims " << miopen::deref(OutputDesc).GetLengths();
        ss << " -F " << ((is_fwd) ? "1" : "2");
        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t
miopenUnsortedSegmentSumForward(miopenHandle_t handle,
                                const miopenTensorDescriptor_t InputDesc,
                                const void* Input,
                                const miopenTensorDescriptor_t OutputDesc,
                                void* Output,
                                const miopenTensorDescriptor_t SegmentIdsDesc,
                                const void* segment_ids)
{
    MIOPEN_LOG_FUNCTION(handle, InputDesc, OutputDesc, SegmentIdsDesc);
    LogCmdUnsortedSegmentSum(InputDesc, OutputDesc, true);
    return miopen::try_([&] {
        miopen::UnsortedSegmentSum::UnsortedSegmentSumForward(miopen::deref(handle),
                                                              miopen::deref(InputDesc),
                                                              DataCast(Input),
                                                              miopen::deref(OutputDesc),
                                                              DataCast(Output),
                                                              miopen::deref(SegmentIdsDesc),
                                                              DataCast(segment_ids));
    });
}

extern "C" miopenStatus_t
miopenUnsortedSegmentSumBackward(miopenHandle_t handle,
                                 const miopenTensorDescriptor_t OutputGradDesc,
                                 const void* OutputGrad,
                                 const miopenTensorDescriptor_t InputGradDesc,
                                 void* InputGrad,
                                 const miopenTensorDescriptor_t SegmentIdsDesc,
                                 const void* segment_ids)
{
    MIOPEN_LOG_FUNCTION(handle, InputGradDesc, OutputGradDesc, SegmentIdsDesc);
    LogCmdUnsortedSegmentSum(InputGradDesc, OutputGradDesc, false);
    return miopen::try_([&] {
        miopen::UnsortedSegmentSum::UnsortedSegmentSumBackward(miopen::deref(handle),
                                                               miopen::deref(OutputGradDesc),
                                                               DataCast(OutputGrad),
                                                               miopen::deref(InputGradDesc),
                                                               DataCast(InputGrad),
                                                               miopen::deref(SegmentIdsDesc),
                                                               DataCast(segment_ids));
    });
}
