/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#include <miopen/check_numerics.hpp>
#include <miopen/env.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor.hpp>
#include <miopen/datatype.hpp>

MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_CHECK_NUMERICS)

#define HIP_CHECK(status)                                                                      \
    do                                                                                         \
    {                                                                                          \
        if(status != hipSuccess)                                                               \
        {                                                                                      \
            std::cerr << "HIP Error: " << hipGetErrorString(status) << " in file " << __FILE__ \
                      << " at line " << __LINE__ << std::endl;                                 \
            exit(EXIT_FAILURE);                                                                \
        }                                                                                      \
    } while(0)

namespace miopen {

bool CheckNumericsEnabled(const int bitMask)
{
    return (env::value(MIOPEN_CHECK_NUMERICS) & bitMask) != 0;
}

// Must keep this structure synchronized with one in MIOpenCheckNumerics
struct CheckNumericsResult
{
    float sum    = 0.0f;
    float absSum = 0.0f;
    float min    = 0.0f;
    float max    = 0.0f;

    int hasZero = 0;
    int hasNan  = 0;
    int hasInf  = 0;
};

struct CallbackData
{
    CheckNumericsResult *abnormal;
    int mode;
    bool isInput;
    int numElements;
    ConstData_t ptr;
    std::string tensorStr;
};

std::string GetKernelName(miopenDataType_t data_type)
{
    switch(data_type)
    {
    case miopenFloat: return {"check_numerics_fp32"};
    case miopenHalf: return {"check_numerics_fp16"};
    case miopenBFloat16: return {"check_numerics_bf16"};
    case miopenFloat8_fnuz: return {"check_numerics_fp8"};
    case miopenBFloat8_fnuz: return {"check_numerics_bf8"};
    case miopenInt64:
    case miopenInt32:
    case miopenInt8:
    case miopenDouble:
    default: return {""};
    }
}

void initCheckNumericsResult(void* args)
{
    CheckNumericsResult h_args{*(reinterpret_cast<CheckNumericsResult*>(args))};
    h_args.sum    = 0.0f;
    h_args.absSum = 0.0f;
    h_args.min    = 0.0f;
    h_args.max    = 0.0f;

    h_args.hasZero = 0;
    h_args.hasNan  = 0;
    h_args.hasInf  = 0;
}

void checkNumericsCallback(void *data)
{
    CallbackData *cd = static_cast<CallbackData *>(data);

    auto *abnormal_h = cd->abnormal;
    auto mode = cd->mode;
    auto isInput = cd->isInput;
    auto numElements = cd->numElements;
    auto ptr = cd->ptr;
    std::string tensorStr = cd->tensorStr;

    const int computeStats = (mode & CheckNumerics::ComputeStats);

    bool isAbnormal = (abnormal_h->hasNan != 0) || (abnormal_h->hasInf != 0);

    if(((mode & CheckNumerics::Info) != 0) || (((mode & CheckNumerics::Warn) != 0) && isAbnormal))
    {
        MIOPEN_LOG((isAbnormal ? miopen::LoggingLevel::Warning : miopen::LoggingLevel::Info),
                   (isInput ? "INPUT " : "OUTPUT")
                       << " ptr=" << ptr << " zeros=" << abnormal_h->hasZero
                       << " nans=" << abnormal_h->hasNan << " infs=" << abnormal_h->hasInf << "  {"
                       << tensorStr << "}");
        if(computeStats != 0)
        {
            assert(numElements != 0);
            MIOPEN_LOG((isAbnormal ? miopen::LoggingLevel::Warning : miopen::LoggingLevel::Info),
                       "Stats: mean=" << (abnormal_h->sum / numElements)
                                      << " absmean=" << (abnormal_h->absSum / numElements)
                                      << " min=" << abnormal_h->min << " max=" << abnormal_h->max);
        }
    }
}

bool checkNumericsImpl(
    const Handle& handle, int mode, const TensorDescriptor& dDesc, ConstData_t data, bool isInput)
{
    int numElements = dDesc.GetElementSize();
    static CheckNumericsResult abnormal_h; // TODO - this can be static for now since we are only checking one stream at a time
    auto abnormal_d =
        handle.CreateAsync(sizeof(CheckNumericsResult)); // TODO - someday avoid slow malloc/free here

    // Assign host function to the stream (note that hipMemsetAsync does not appear to work with hip graph)
    HIP_CHECK(hipLaunchHostFunc(handle.GetStream(), initCheckNumericsResult, &abnormal_h));

    HIP_CHECK(hipMemcpyAsync(abnormal_d.get(), &abnormal_h, sizeof(CheckNumericsResult), hipMemcpyHostToDevice, handle.GetStream()));
    const size_t threadsPerBlock = 256;
    const size_t numBlocks       = handle.GetMaxComputeUnits() * 6;
    const int computeStats       = (mode & CheckNumerics::ComputeStats);
    // TODO - some constants we should get from the device:
    std::string program_name      = "MIOpenCheckNumerics.cpp";
    std::string kernel_name       = GetKernelName(dDesc.GetType());
    const std::vector<size_t> vld = {size_t{threadsPerBlock}, size_t{1}, size_t{1}};
    const std::vector<size_t> vgd = {numBlocks, size_t{1}, size_t{1}};
    handle.AddKernel(
        "MIOpenCheckNumerics", "MIOpenCheckNumerics", program_name, kernel_name, vld, vgd, "")(
        data, numElements, abnormal_d.get(), computeStats);

    HIP_CHECK(hipMemcpyAsync(&abnormal_h, abnormal_d.get(), sizeof(CheckNumericsResult), hipMemcpyDeviceToHost, handle.GetStream()));

    CallbackData *callbackData = new CallbackData;
    callbackData->abnormal = &abnormal_h;
    callbackData->mode = mode;
    callbackData->isInput = isInput;
    callbackData->numElements = numElements;
    callbackData->ptr = data;
    std::stringstream tmp;
    tmp << dDesc;
    callbackData->tensorStr = tmp.str();

    HIP_CHECK(hipLaunchHostFunc(handle.GetStream(), checkNumericsCallback, callbackData));
    MIOPEN_LOG(LoggingLevel::Info, "JFL: post 2nd hipLaunchHostFunc");

    hipStreamCaptureStatus captureStatus;
    hipStreamIsCapturing(handle.GetStream(), &captureStatus);
    if (captureStatus == hipStreamCaptureStatusActive)
        return false;

    MIOPEN_LOG(LoggingLevel::Info, "JFL: not capturing hip graph, need to sycnhronize");
    HIP_CHECK(hipStreamSynchronize(handle.GetStream()));

    MIOPEN_LOG(LoggingLevel::Info, "JFL: after captureStatus");
    bool isAbnormal = (abnormal_h.hasNan != 0) || (abnormal_h.hasInf != 0);

    if(isAbnormal)
    {

        if((mode & CheckNumerics::Throw) != 0)
        {
            if(isInput)
            {
                MIOPEN_THROW(miopenStatusInternalError,
                             "abnormal checkNumerics result detected on INPUT");
            }
            else
            {
                MIOPEN_THROW(miopenStatusInternalError,
                             "abnormal checkNumerics result detected on OUTPUT");
            }
        }
        if((mode & CheckNumerics::Abort) != 0)
        {
            abort();
        }
    }

    // TODO - free up allocated memory

    return isAbnormal;
};

// Checks data for input
// Returns: 1 if abnormal value (inf or nan) detected in specified data, 0 otherwise
bool checkNumericsInput(const Handle& handle, const TensorDescriptor& dDesc, ConstData_t data)
{
    return checkNumericsImpl(handle, env::value(MIOPEN_CHECK_NUMERICS), dDesc, data, true);
}

// Synchronizes to wait for kernel to finish, then checks data for output:
// Returns: 1 if abnormal value (inf or nan) detected in specified data, 0 otherwise
bool checkNumericsOutput(const Handle& handle, const TensorDescriptor& dDesc, ConstData_t data)
{
    //handle.Finish();
    return checkNumericsImpl(handle, env::value(MIOPEN_CHECK_NUMERICS), dDesc, data, false);
}

} // namespace miopen
