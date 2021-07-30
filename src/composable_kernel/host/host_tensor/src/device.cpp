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
        hipEventCreate(&mStart);
        hipEventCreate(&mEnd);
    }

    ~KernelTimerImpl()
    {
        hipEventDestroy(mStart);
        hipEventDestroy(mEnd);
    }

    void Start()
    {
        hipDeviceSynchronize();
        hipEventRecord(mStart, 0);
    }

    void End()
    {
        hipEventRecord(mEnd, 0);
        hipEventSynchronize(mEnd);
    }

    float GetElapsedTime() const
    {
        float time;
        hipEventElapsedTime(&time, mStart, mEnd);
        return time;
    }

    hipEvent_t mStart, mEnd;
};

KernelTimer::KernelTimer() : impl(new KernelTimerImpl()) {}

KernelTimer::~KernelTimer() {}

void KernelTimer::Start() { impl->Start(); }

void KernelTimer::End() { impl->End(); }

float KernelTimer::GetElapsedTime() const { return impl->GetElapsedTime(); }
