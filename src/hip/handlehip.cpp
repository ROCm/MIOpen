#include <miopen/handle.hpp>
#include <miopen/errors.hpp>
#include <miopen/device_name.hpp>
#include <miopen/kernel_cache.hpp>
#include <algorithm>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <cassert>
#include <chrono>
#include <thread>

namespace miopen {

hipDevice_t get_device(int id)
{
    hipDevice_t device;
    auto status = hipDeviceGet(&device, id);
    if (status != hipSuccess) MIOPEN_THROW("No device");
    return device;
}

int get_device_id() // Get random device
{
    int device;
    auto status = hipGetDevice(&device);
    if (status != hipSuccess) MIOPEN_THROW("No device");
    return device;
}

void set_device(int id)
{
    auto status = hipSetDevice(id);
    if (status != hipSuccess) MIOPEN_THROW("Error setting device");
}

void set_default_device()
{
    int n;
    auto status = hipGetDeviceCount(&n);
    if (status != hipSuccess) MIOPEN_THROW("Error getting device count");
    // Pick device based on process id
    auto pid = ::getpid();
    assert(pid > 0);
    set_device(pid % n);
}

struct HandleImpl
{
    // typedef MIOPEN_MANAGE_PTR(hipStream_t, hipStreamDestroy) StreamPtr;
    using StreamPtr = std::shared_ptr<typename std::remove_pointer<hipStream_t>::type>;

    HandleImpl()
    {}

    StreamPtr create_stream()
    {
        hipStream_t result;
        auto status = hipStreamCreate(&result);
        if (status != hipSuccess) MIOPEN_THROW_HIP_STATUS(status, "Failed to allocate stream");
        return StreamPtr{result, &hipStreamDestroy};
    }

    static StreamPtr reference_stream(hipStream_t s)
    {
        return StreamPtr{s, null_deleter{}};
    }

    void elapsed_time(hipEvent_t start, hipEvent_t stop)
    {
        hipEventElapsedTime(&this->profiling_result, start, stop);
    }

    std::function<void(hipEvent_t, hipEvent_t)> elapsed_time_handler()
    {
        return std::bind(&HandleImpl::elapsed_time, this, std::placeholders::_1, std::placeholders::_2);
    }

    bool enable_profiling = false;
    StreamPtr stream;
    float profiling_result = 0.0;
    KernelCache cache;
};

Handle::Handle (miopenAcceleratorQueue_t stream) 
: impl(new HandleImpl())
{
    if (stream == nullptr) this->impl->stream = HandleImpl::reference_stream(nullptr);
    else this->impl->stream = HandleImpl::reference_stream(stream);
}

Handle::Handle ()
: impl(new HandleImpl())
{
    set_default_device();
    // this->impl->stream = impl->create_stream();
    this->impl->stream = HandleImpl::reference_stream(nullptr);
}

Handle::~Handle() {}

void Handle::SetStream(miopenAcceleratorQueue_t streamID) const
{
    this->impl->stream = HandleImpl::reference_stream(streamID);
}

miopenAcceleratorQueue_t Handle::GetStream() const
{
    return impl->stream.get();
}

void Handle::EnableProfiling(bool enable)
{
    this->impl->enable_profiling = enable;
}

float Handle::GetKernelTime() const
{
    return this->impl->profiling_result;
}

std::size_t GetAvailableMemory()
{
    size_t free, total;
    auto status = hipMemGetInfo(&free, &total);
    if (status != hipSuccess) MIOPEN_THROW_HIP_STATUS(status, "Failed getting available memory");
    return free;
}

ManageDataPtr Handle::Create(std::size_t sz)
{
    this->Finish();
    if (sz > GetAvailableMemory()) MIOPEN_THROW("Memory not available to allocate buffer: " + std::to_string(sz));
    void * result;
    auto status = hipMalloc(&result, sz);
    if (status != hipSuccess)
    {
        status = hipHostMalloc(&result, sz);
        if (status != hipSuccess) MIOPEN_THROW_HIP_STATUS(status, "Hip error creating buffer " + std::to_string(sz) + ": ");
    }
    return ManageDataPtr{result};
}
ManageDataPtr& Handle::WriteTo(const void* data, ManageDataPtr& ddata, std::size_t sz)
{
    this->Finish();
    auto status = hipMemcpy(ddata.get(), data, sz, hipMemcpyHostToDevice);
    if (status != hipSuccess) MIOPEN_THROW_HIP_STATUS(status, "Hip error writing to buffer: ");
    return ddata;
}
void Handle::ReadTo(void* data, const ManageDataPtr& ddata, std::size_t sz)
{
    this->Finish();
    auto status = hipMemcpy(data, ddata.get(), sz, hipMemcpyDeviceToHost);
    if (status != hipSuccess) MIOPEN_THROW_HIP_STATUS(status, "Hip error reading from buffer: ");
}

void Handle::Copy(ConstData_t src, Data_t dest, std::size_t size)
{
    auto status = hipMemcpy(dest, src, size, hipMemcpyDeviceToDevice);
    if (status != hipSuccess) MIOPEN_THROW_HIP_STATUS(status, "Hip error copying buffer: ");
}

KernelInvoke Handle::GetKernel(
        const std::string& algorithm,
        const std::string& network_config,
        const std::string& program_name,
        const std::string& kernel_name,
        const std::vector<size_t>& vld,
        const std::vector<size_t>& vgd,
        const std::string& params)
{
    auto k = this->impl->cache.GetKernel(*this,
            algorithm,
            network_config,
            program_name,
            kernel_name,
            vld,
            vgd,
            params);
    if (this->impl->enable_profiling) return k.Invoke(this->GetStream(), this->impl->elapsed_time_handler());
    else return k.Invoke(this->GetStream());
}

KernelInvoke Handle::GetKernel(
    const std::string& algorithm,
    const std::string& network_config)
{
    auto k = this->impl->cache.GetKernel(
            algorithm,
            network_config);
    if (this->impl->enable_profiling) return k.Invoke(this->GetStream(), this->impl->elapsed_time_handler());
    else return k.Invoke(this->GetStream());
}

Program Handle::LoadProgram(const std::string &program_name, std::string params, bool is_kernel_str)
{
    params += " -mcpu=" + this->GetDeviceName();
    return HIPOCProgram{program_name, params, is_kernel_str};
}

void Handle::Finish() const
{
#if MIOPEN_BUILD_DEV
    auto start = std::chrono::system_clock::now();
    auto ev = make_hip_event();
    hipEventRecord(ev.get(), this->GetStream());
    while(hipEventQuery(ev.get()) == hipErrorNotReady)
    {
        std::this_thread::yield();
        if ((std::chrono::system_clock::now()-start) > std::chrono::seconds(60)) 
        {
            std::cerr << "Timeout: Handle::Finish" << std::endl;
            std::abort();
        }
    }
#else
    auto status = hipStreamSynchronize(this->GetStream());
    if (status != hipSuccess) MIOPEN_THROW_HIP_STATUS(status, "Failed hip sychronization");
#endif
}
void Handle::Flush() const
{

}

bool Handle::IsProfilingEnabled() const
{
	return this->impl->enable_profiling;
}

void Handle::ResetKernelTime()
{
    this->impl->profiling_result = 0.0;
}
void Handle::AccumKernelTime(float x)
{
    this->impl->profiling_result += x;
}

std::size_t Handle::GetLocalMemorySize()
{
    int result;
    auto status = hipDeviceGetAttribute(&result, hipDeviceAttributeMaxSharedMemoryPerBlock, get_device_id());
    if (status != hipSuccess) MIOPEN_THROW_HIP_STATUS(status);

    return result;
}

std::size_t Handle::GetMaxComputeUnits()
{
    int result;
    auto status = hipDeviceGetAttribute(&result, hipDeviceAttributeMultiprocessorCount, get_device_id());
    if (status != hipSuccess) MIOPEN_THROW_HIP_STATUS(status);

    return result;
}

std::string Handle::GetDeviceName()
{
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, get_device_id());
    std::string n("gfx"+std::to_string(props.gcnArch));
	return GetDeviceNameFromMap(n);
}
} // namespace miopen
