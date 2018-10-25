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
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>

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
    }
    return "Unknown error status";
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
