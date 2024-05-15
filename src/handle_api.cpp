/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#include <cstdio>
#include <miopen/version.h>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>

extern const char* miopenTensorArgumentToString(miopenTensorArgumentId_t arg)
{
    switch(arg)
    {
    case miopenTensorArgumentIdInvalid: return "miopenTensorArgumentIdInvalid";

    case miopenTensorConvolutionX: return "miopenTensorConvolutionX";

    case miopenTensorConvolutionW: return "miopenTensorConvolutionW";

    case miopenTensorConvolutionY: return "miopenTensorConvolutionY";

    case miopenTensorMhaK: return "miopenTensorMhaK";

    case miopenTensorMhaQ: return "miopenTensorMhaQ";

    case miopenTensorMhaV: return "miopenTensorMhaV";

    case miopenTensorMhaDescaleK: return "miopenTensorMhaDescaleK";

    case miopenTensorMhaDescaleQ: return "miopenTensorMhaDescaleQ";

    case miopenTensorMhaDescaleV: return "miopenTensorMhaDescaleV";

    case miopenTensorMhaDescaleS: return "miopenTensorMhaDescaleS";

    case miopenTensorMhaScaleS: return "miopenTensorMhaScaleS";

    case miopenTensorMhaScaleO: return "miopenTensorMhaScaleO";

    case miopenTensorMhaDropoutProbability: return "miopenTensorMhaDropoutProbability";

    case miopenTensorMhaDropoutSeed: return "miopenTensorMhaDropoutSeed";

    case miopenTensorMhaDropoutOffset: return "miopenTensorMhaDropoutOffset";

    case miopenTensorMhaO: return "miopenTensorMhaO";

    case miopenTensorMhaAmaxO: return "miopenTensorMhaAmaxO";

    case miopenTensorMhaAmaxS: return "miopenTensorMhaAmaxS";

    case miopenTensorMhaM: return "miopenTensorMhaM";

    case miopenTensorMhaZInv: return "miopenTensorMhaZInv";

    case miopenTensorMhaDO: return "miopenTensorMhaDO";

    case miopenTensorMhaDescaleO: return "miopenTensorMhaDescaleO";

    case miopenTensorMhaDescaleDO: return "miopenTensorMhaDescaleDO";

    case miopenTensorMhaDescaleDS: return "miopenTensorMhaDescaleDS";

    case miopenTensorMhaScaleDS: return "miopenTensorMhaScaleDS";

    case miopenTensorMhaScaleDQ: return "miopenTensorMhaScaleDQ";

    case miopenTensorMhaScaleDK: return "miopenTensorMhaScaleDK";

    case miopenTensorMhaScaleDV: return "miopenTensorMhaScaleDV";

    case miopenTensorMhaDQ: return "miopenTensorMhaDQ";

    case miopenTensorMhaDK: return "miopenTensorMhaDK";

    case miopenTensorMhaDV: return "miopenTensorMhaDV";

    case miopenTensorMhaAmaxDQ: return "miopenTensorMhaAmaxDQ";

    case miopenTensorMhaAmaxDK: return "miopenTensorMhaAmaxDK";

    case miopenTensorMhaAmaxDV: return "miopenTensorMhaAmaxDV";

    case miopenTensorMhaAmaxDS: return "miopenTensorMhaAmaxDS";

#ifdef MIOPEN_BETA_API

    case miopenTensorActivationX: return "miopenTensorActivationX";

    case miopenTensorActivationY: return "miopenTensorActivationY";

    case miopenTensorActivationDX: return "miopenTensorActivationDX";

    case miopenTensorActivationDY: return "miopenTensorActivationDY";

    case miopenTensorBiasX: return "miopenTensorBiasX";

    case miopenTensorBiasY: return "miopenTensorBiasY";

    case miopenTensorBias: return "miopenTensorBias";

    case miopenTensorSoftmaxX: return "miopenTensorSoftmaxX";

    case miopenTensorSoftmaxY: return "miopenTensorSoftmaxY";

    case miopenTensorSoftmaxDX: return "miopenTensorSoftmaxDX";

    case miopenTensorSoftmaxDY: return "miopenTensorSoftmaxDY";

    case miopenTensorBatchnormX: return "miopenTensorBatchnormX";

    case miopenTensorBatchnormY: return "miopenTensorBatchnormY";

    case miopenTensorBatchnormRunningMean: return "miopenTensorBatchnormRunningMean";

    case miopenTensorBatchnormRunningVariance: return "miopenTensorBatchnormRunningVariance";

    case miopenTensorBatchnormSavedMean: return "miopenTensorBatchnormSavedMean";

    case miopenTensorBatchnormSavedVariance: return "miopenTensorBatchnormSavedVariance";

    case miopenTensorBatchnormScale: return "miopenTensorBatchnormScale";

    case miopenTensorBatchnormScaleDiff: return "miopenTensorBatchnormScaleDiff";

    case miopenTensorBatchnormEstimatedMean: return "miopenTensorBatchnormEstimatedMean";

    case miopenTensorBatchnormEstimatedVariance: return "miopenTensorBatchnormEstimatedVariance";

    case miopenTensorBatchnormBias: return "miopenTensorBatchnormBias";

    case miopenTensorBatchnormBiasDiff: return "miopenTensorBatchnormBiasDiff";

    case miopenTensorBatchnormDX: return "miopenTensorBatchnormDX";

    case miopenTensorBatchnormDY: return "miopenTensorBatchnormDY";

#endif

    case miopenTensorArgumentIsScalar: return "miopenTensorArgumentIsScalar";

#ifdef MIOPEN_BETA_API

    case miopenScalarBatchnormExpAvgFactor: return "miopenScalarBatchnormExpAvgFactor";

    case miopenScalarBatchnormEpsilon: return "miopenScalarBatchnormEpsilon";

#endif
    }
}

extern "C" const char* miopenGetErrorString(miopenStatus_t error)
{
    switch(error)
    {
    case miopenStatusSuccess: return "miopenStatusSuccess";

    case miopenStatusNotInitialized: return "miopenStatusNotInitialized";

    case miopenStatusInvalidValue: return "miopenStatusInvalidValue";

    case miopenStatusBadParm: return "miopenStatusBadParm";

    case miopenStatusAllocFailed: return "miopenStatusAllocFailed";

    case miopenStatusInternalError: return "miopenStatusInternalError";

    case miopenStatusNotImplemented: return "miopenStatusNotImplemented";

    case miopenStatusUnknownError: return "miopenStatusUnknownError";

    case miopenStatusUnsupportedOp: return "miopenStatusUnsupportedOp";

    case miopenStatusGpuOperationsSkipped: return "miopenStatusGpuOperationsSkipped";

    case miopenStatusVersionMismatch: return "miopenStatusVersionMismatch";
    }
    return "Unknown error status";
}

extern "C" miopenStatus_t miopenGetVersion(size_t* major, size_t* minor, size_t* patch)
{
    return miopen::try_([&] {
        if(major != nullptr)
            *major = MIOPEN_VERSION_MAJOR;
        if(minor != nullptr)
            *minor = MIOPEN_VERSION_MINOR;
        if(patch != nullptr)
            *patch = MIOPEN_VERSION_PATCH;
    });
}

extern "C" miopenStatus_t miopenCreate(miopenHandle_t* handle)
{

    return miopen::try_([&] { miopen::deref(handle) = new miopen::Handle(); });
}

extern "C" miopenStatus_t miopenCreateWithStream(miopenHandle_t* handle,
                                                 miopenAcceleratorQueue_t stream)
{

    return miopen::try_([&] { miopen::deref(handle) = new miopen::Handle(stream); });
}

extern "C" miopenStatus_t miopenSetStream(miopenHandle_t handle, miopenAcceleratorQueue_t streamID)
{
    return miopen::try_([&] { miopen::deref(handle).SetStream(streamID); });
}

extern "C" miopenStatus_t miopenGetStream(miopenHandle_t handle, miopenAcceleratorQueue_t* streamID)
{
    return miopen::try_([&] { miopen::deref(streamID) = miopen::deref(handle).GetStream(); });
}

extern "C" miopenStatus_t miopenSetAllocator(miopenHandle_t handle,
                                             miopenAllocatorFunction allocator,
                                             miopenDeallocatorFunction deallocator,
                                             void* allocatorContext)
{
    return miopen::try_(
        [&] { miopen::deref(handle).SetAllocator(allocator, deallocator, allocatorContext); });
}

extern "C" miopenStatus_t miopenDestroy(miopenHandle_t handle)
{
    return miopen::try_([&] { miopen_destroy_object(handle); });
}

extern "C" miopenStatus_t miopenGetKernelTime(miopenHandle_t handle, float* time)
{
    return miopen::try_([&] { miopen::deref(time) = miopen::deref(handle).GetKernelTime(); });
}
extern "C" miopenStatus_t miopenEnableProfiling(miopenHandle_t handle, bool enable)
{
    return miopen::try_([&] { miopen::deref(handle).EnableProfiling(enable); });
}
