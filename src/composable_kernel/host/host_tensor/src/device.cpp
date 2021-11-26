/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
#include "device.hpp"

DeviceMem::DeviceMem(std::size_t mem_size) : mMemSize(mem_size)
{
    hipGetErrorString(hipMalloc(static_cast<void**>(&mpDeviceBuf), mMemSize));
}

void* DeviceMem::GetDeviceBuffer() { return mpDeviceBuf; }

void DeviceMem::ToDevice(const void* p)
{
    hipGetErrorString(
        hipMemcpy(mpDeviceBuf, const_cast<void*>(p), mMemSize, hipMemcpyHostToDevice));
}

void DeviceMem::FromDevice(void* p)
{
    hipGetErrorString(hipMemcpy(p, mpDeviceBuf, mMemSize, hipMemcpyDeviceToHost));
}

DeviceMem::~DeviceMem() { hipGetErrorString(hipFree(mpDeviceBuf)); }

struct KernelTimerImpl
{
    KernelTimerImpl()
    {
        hipGetErrorString(hipEventCreate(&mStart));
        hipGetErrorString(hipEventCreate(&mEnd));
    }

    ~KernelTimerImpl()
    {
        hipGetErrorString(hipEventDestroy(mStart));
        hipGetErrorString(hipEventDestroy(mEnd));
    }

    void Start()
    {
        hipGetErrorString(hipDeviceSynchronize());
        hipGetErrorString(hipEventRecord(mStart, nullptr));
    }

    void End()
    {
        hipGetErrorString(hipEventRecord(mEnd, nullptr));
        hipGetErrorString(hipEventSynchronize(mEnd));
    }

    float GetElapsedTime() const
    {
        float time;
        hipGetErrorString(hipEventElapsedTime(&time, mStart, mEnd));
        return time;
    }

    hipEvent_t mStart, mEnd;
};

KernelTimer::KernelTimer() : impl(new KernelTimerImpl()) {}

KernelTimer::~KernelTimer() {}

void KernelTimer::Start() { impl->Start(); }

void KernelTimer::End() { impl->End(); }

float KernelTimer::GetElapsedTime() const { return impl->GetElapsedTime(); }
