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
#include <algorithm>
#include <miopen/device_name.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/binary_cache.hpp>
#include <boost/filesystem.hpp>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <cassert>
#include <chrono>
#include <thread>

namespace miopen {

// Get current context
// We leak resources for now as there is no hipCtxRetain API
hipCtx_t get_ctx()
{
    hipInit(0);
    hipCtx_t ctx;
    auto status = hipCtxGetCurrent(&ctx);
    if(status != hipSuccess)
        MIOPEN_THROW("No device");
    return ctx;
}

std::size_t GetAvailableMemory()
{
    size_t free, total;
    auto status = hipMemGetInfo(&free, &total);
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "Failed getting available memory");
    return free;
}

void* default_allocator(void*, size_t sz)
{
    if(sz > GetAvailableMemory())
        MIOPEN_THROW("Memory not available to allocate buffer: " + std::to_string(sz));
    void* result;
    auto status = hipMalloc(&result, sz);
    if(status != hipSuccess)
    {
        status = hipHostMalloc(&result, sz);
        if(status != hipSuccess)
            MIOPEN_THROW_HIP_STATUS(status,
                                    "Hip error creating buffer " + std::to_string(sz) + ": ");
    }
    return result;
}

void default_deallocator(void*, void* mem) { hipFree(mem); }

int get_device_id() // Get random device
{
    int device;
    auto status = hipGetDevice(&device);
    if(status != hipSuccess)
        MIOPEN_THROW("No device");
    return device;
}

void set_device(int id)
{
    auto status = hipSetDevice(id);
    if(status != hipSuccess)
        MIOPEN_THROW("Error setting device");
}

void set_ctx(hipCtx_t ctx)
{
    auto status = hipCtxSetCurrent(ctx);
    if(status != hipSuccess)
        MIOPEN_THROW("Error setting context");
}

int set_default_device()
{
    int n;
    auto status = hipGetDeviceCount(&n);
    if(status != hipSuccess)
        MIOPEN_THROW("Error getting device count");
    // Pick device based on process id
    auto pid = ::getpid();
    assert(pid > 0);
    set_device(pid % n);
    return (pid % n);
}

struct HandleImpl
{
    // typedef MIOPEN_MANAGE_PTR(hipStream_t, hipStreamDestroy) StreamPtr;
    using StreamPtr = std::shared_ptr<typename std::remove_pointer<hipStream_t>::type>;

    HandleImpl() : ctx(get_ctx()) {}

    StreamPtr create_stream()
    {
        hipStream_t result;
        auto status = hipStreamCreate(&result);
        if(status != hipSuccess)
            MIOPEN_THROW_HIP_STATUS(status, "Failed to allocate stream");
        return StreamPtr{result, &hipStreamDestroy};
    }

    static StreamPtr reference_stream(hipStream_t s) { return StreamPtr{s, null_deleter{}}; }

    void elapsed_time(hipEvent_t start, hipEvent_t stop)
    {
        hipEventElapsedTime(&this->profiling_result, start, stop);
    }

    std::function<void(hipEvent_t, hipEvent_t)> elapsed_time_handler()
    {
        return std::bind(
            &HandleImpl::elapsed_time, this, std::placeholders::_1, std::placeholders::_2);
    }

    void set_ctx()
    {
        miopen::set_ctx(this->ctx);
        // miopen::set_device(this->device);
        // TODO: Check device matches
    }

    bool enable_profiling  = false;
    StreamPtr stream       = nullptr;
    float profiling_result = 0.0;
    int device             = -1;
    Allocator allocator{};
    KernelCache cache;
    hipCtx_t ctx;
};

Handle::Handle(miopenAcceleratorQueue_t stream) : impl(new HandleImpl())
{
    this->impl->device = get_device_id();
    this->impl->ctx    = get_ctx();

    if(stream == nullptr)
        this->impl->stream = HandleImpl::reference_stream(nullptr);
    else
        this->impl->stream = HandleImpl::reference_stream(stream);

    this->SetAllocator(nullptr, nullptr, nullptr);
}

Handle::Handle() : impl(new HandleImpl())
{
#if MIOPEN_BUILD_DEV
    this->impl->device = set_default_device();
    this->impl->ctx    = get_ctx();
    this->impl->stream = impl->create_stream();
#else
    this->impl->device = get_device_id();
    this->impl->ctx    = get_ctx();
    this->impl->stream = HandleImpl::reference_stream(nullptr);
#endif
    this->SetAllocator(nullptr, nullptr, nullptr);
}

Handle::~Handle() {}

void Handle::SetStream(miopenAcceleratorQueue_t streamID) const
{
    this->impl->stream = HandleImpl::reference_stream(streamID);
}

miopenAcceleratorQueue_t Handle::GetStream() const { return impl->stream.get(); }

void Handle::SetAllocator(miopenAllocatorFunction allocator,
                          miopenDeallocatorFunction deallocator,
                          void* allocatorContext) const
{
    this->impl->allocator.allocator   = allocator == nullptr ? default_allocator : allocator;
    this->impl->allocator.deallocator = deallocator == nullptr ? default_deallocator : deallocator;

    this->impl->allocator.context = allocatorContext;
}

void Handle::EnableProfiling(bool enable) { this->impl->enable_profiling = enable; }

float Handle::GetKernelTime() const { return this->impl->profiling_result; }

Allocator::ManageDataPtr Handle::Create(std::size_t sz)
{
    this->Finish();
    return this->impl->allocator(sz);
}
Allocator::ManageDataPtr&
Handle::WriteTo(const void* data, Allocator::ManageDataPtr& ddata, std::size_t sz)
{
    this->Finish();
    auto status = hipMemcpy(ddata.get(), data, sz, hipMemcpyHostToDevice);
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "Hip error writing to buffer: ");
    return ddata;
}
void Handle::ReadTo(void* data, const Allocator::ManageDataPtr& ddata, std::size_t sz)
{
    this->Finish();
    auto status = hipMemcpy(data, ddata.get(), sz, hipMemcpyDeviceToHost);
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "Hip error reading from buffer: ");
}

void Handle::Copy(ConstData_t src, Data_t dest, std::size_t size)
{
    this->impl->set_ctx();
    auto status = hipMemcpy(dest, src, size, hipMemcpyDeviceToDevice);
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "Hip error copying buffer: ");
}

KernelInvoke Handle::GetKernel(const std::string& algorithm,
                               const std::string& network_config,
                               const std::string& program_name,
                               const std::string& kernel_name,
                               const std::vector<size_t>& vld,
                               const std::vector<size_t>& vgd,
                               const std::string& params)
{
    this->impl->set_ctx();
    auto k = this->impl->cache.GetKernel(
        *this, algorithm, network_config, program_name, kernel_name, vld, vgd, params);
    if(this->impl->enable_profiling)
        return k.Invoke(this->GetStream(), this->impl->elapsed_time_handler());
    else
        return k.Invoke(this->GetStream());
}

KernelInvoke Handle::GetKernel(const std::string& algorithm, const std::string& network_config)
{
    this->impl->set_ctx();
    auto k = this->impl->cache.GetKernel(algorithm, network_config);
    if(this->impl->enable_profiling)
        return k.Invoke(this->GetStream(), this->impl->elapsed_time_handler());
    else
        return k.Invoke(this->GetStream());
}

Program Handle::LoadProgram(const std::string& program_name, std::string params, bool is_kernel_str)
{
    this->impl->set_ctx();
    params += " -mcpu=" + this->GetDeviceName();
    auto cache_file =
        miopen::LoadBinary(this->GetDeviceName(), program_name, params, is_kernel_str);
    if(cache_file.empty())
    {
        auto p = HIPOCProgram{program_name, params, is_kernel_str};

        // Save to cache
        auto path = miopen::GetCachePath() / boost::filesystem::unique_path();
        boost::filesystem::copy_file(p.GetBinary(), path);
        miopen::SaveBinary(path, this->GetDeviceName(), program_name, params, is_kernel_str);

        return p;
    }
    else
    {
        return HIPOCProgram{program_name, cache_file};
    }
}

void Handle::Finish() const
{
    this->impl->set_ctx();
#if 0
    auto start = std::chrono::system_clock::now();
    auto ev    = make_hip_event();
    hipEventRecord(ev.get(), this->GetStream());
    while(hipEventQuery(ev.get()) == hipErrorNotReady)
    {
        std::this_thread::yield();
        if((std::chrono::system_clock::now() - start) > std::chrono::seconds(60))
        {
            std::cerr << "Timeout: Handle::Finish" << std::endl;
            std::abort();
        }
    }
#else
    // hipStreamSynchronize is broken, so we use hipEventSynchronize instead
    auto ev = make_hip_event();
    hipEventRecord(ev.get(), this->GetStream());
    auto status = hipEventSynchronize(ev.get());
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "Failed hip sychronization");
#endif
}
void Handle::Flush() const {}

bool Handle::IsProfilingEnabled() const { return this->impl->enable_profiling; }

void Handle::ResetKernelTime() { this->impl->profiling_result = 0.0; }
void Handle::AccumKernelTime(float curr_time) { this->impl->profiling_result += curr_time; }

std::size_t Handle::GetLocalMemorySize()
{
    int result;
    auto status = hipDeviceGetAttribute(
        &result, hipDeviceAttributeMaxSharedMemoryPerBlock, this->impl->device);
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status);

    return result;
}

std::size_t Handle::GetMaxComputeUnits()
{
    int result;
    auto status =
        hipDeviceGetAttribute(&result, hipDeviceAttributeMultiprocessorCount, this->impl->device);
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status);

    return result;
}

std::string Handle::GetDeviceName()
{
    hipDeviceProp_t props{};
    hipGetDeviceProperties(&props, this->impl->device);
    std::string n("gfx" + std::to_string(props.gcnArch));
    return GetDeviceNameFromMap(n);
}

shared<Data_t> Handle::CreateSubBuffer(Data_t data, std::size_t offset, std::size_t)
{
    auto cdata = reinterpret_cast<char*>(data);
    return {cdata + offset, null_deleter{}};
}

shared<ConstData_t> Handle::CreateSubBuffer(ConstData_t data, std::size_t offset, std::size_t)
{
    auto cdata = reinterpret_cast<const char*>(data);
    return {cdata + offset, null_deleter{}};
}
} // namespace miopen
