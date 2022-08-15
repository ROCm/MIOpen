/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017-2020 Advanced Micro Devices, Inc.
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

#include <miopen/config.h>
#include <miopen/handle.hpp>

#include <miopen/binary_cache.hpp>
#include <miopen/env.hpp>
#include <miopen/errors.hpp>
#include <miopen/gemm_geometry.hpp>
#include <miopen/handle_lock.hpp>
#include <miopen/invoker.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/logger.hpp>
#include <miopen/rocm_features.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/timer.hpp>

#if !MIOPEN_ENABLE_SQLITE_KERN_CACHE
#include <miopen/write_file.hpp>
#endif

#include <boost/filesystem.hpp>
#include <miopen/load_file.hpp>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <thread>

#define MIOPEN_WORKAROUND_ROCM_COMPILER_SUPPORT_ISSUE_30 \
    (MIOPEN_USE_COMGR && BUILD_SHARED_LIBS && (HIP_PACKAGE_VERSION_FLAT < 4003000000ULL))

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEVICE_CU)

namespace miopen {

namespace {

#if MIOPEN_WORKAROUND_ROCM_COMPILER_SUPPORT_ISSUE_30
void toCallHipInit() __attribute__((constructor(1000)));
void toCallHipInit() { hipInit(0); }
#endif

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

#if MIOPEN_BUILD_DEV
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
#endif

} // namespace

struct HandleImpl
{
    // typedef MIOPEN_MANAGE_PTR(hipStream_t, hipStreamDestroy) StreamPtr;
    using StreamPtr = std::shared_ptr<typename std::remove_pointer<hipStream_t>::type>;

    HandleImpl() { hipInit(0); }

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
        if(enable_profiling)
            hipEventElapsedTime(&this->profiling_result, start, stop);
    }

    std::function<void(hipEvent_t, hipEvent_t)> elapsed_time_handler()
    {
        return std::bind(
            &HandleImpl::elapsed_time, this, std::placeholders::_1, std::placeholders::_2);
    }

    void set_ctx() const { miopen::set_device(this->device); }

    std::string get_device_name() const
    {
        hipDeviceProp_t props{};
        hipGetDeviceProperties(&props, device);
#if ROCM_FEATURE_HIP_GCNARCHNAME_RETURNS_CODENAME
        const std::string name("gfx" + std::to_string(props.gcnArch));
#else
        const std::string name(props.gcnArchName);
#endif
        MIOPEN_LOG_NQI("Raw device name: " << name);
        return name; // NOLINT (performance-no-automatic-move)
    }

    bool enable_profiling  = false;
    StreamPtr stream       = nullptr;
    float profiling_result = 0.0;
    int device             = -1;
    Allocator allocator{};
    KernelCache cache;
    TargetProperties target_properties;
};

Handle::Handle(miopenAcceleratorQueue_t stream) : impl(std::make_unique<HandleImpl>())
{
    this->impl->device = get_device_id();

    if(stream == nullptr)
        this->impl->stream = HandleImpl::reference_stream(nullptr);
    else
        this->impl->stream = HandleImpl::reference_stream(stream);

    this->SetAllocator(nullptr, nullptr, nullptr);

#if MIOPEN_USE_ROCBLAS
    rhandle_ = CreateRocblasHandle();
#endif
    this->impl->target_properties.Init(this);
    MIOPEN_LOG_NQI(*this);
}

Handle::Handle() : impl(std::make_unique<HandleImpl>())
{
#if MIOPEN_BUILD_DEV
    this->impl->device = set_default_device();
    this->impl->stream = impl->create_stream();
#else
    this->impl->device = get_device_id();
    this->impl->stream = HandleImpl::reference_stream(nullptr);
#endif
    this->SetAllocator(nullptr, nullptr, nullptr);

#if MIOPEN_USE_ROCBLAS
    rhandle_ = CreateRocblasHandle();
#endif
    this->impl->target_properties.Init(this);
    MIOPEN_LOG_NQI(*this);
}

Handle::~Handle() {}

void Handle::SetStream(miopenAcceleratorQueue_t streamID) const
{
    this->impl->stream = HandleImpl::reference_stream(streamID);

#if MIOPEN_USE_ROCBLAS
    rocblas_set_stream(this->rhandle_.get(), this->GetStream());
#endif
    this->impl->target_properties.Init(this);
    MIOPEN_LOG_NQI(*this);
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

void Handle::EnableProfiling(bool enable) const { this->impl->enable_profiling = enable; }

float Handle::GetKernelTime() const { return this->impl->profiling_result; }

Allocator::ManageDataPtr Handle::Create(std::size_t sz) const
{
    MIOPEN_HANDLE_LOCK
    this->Finish();
    return this->impl->allocator(sz);
}

Allocator::ManageDataPtr&
Handle::WriteTo(const void* data, Allocator::ManageDataPtr& ddata, std::size_t sz) const
{
    MIOPEN_HANDLE_LOCK
    this->Finish();
    auto status = hipMemcpy(ddata.get(), data, sz, hipMemcpyHostToDevice);
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "Hip error writing to buffer: ");
    return ddata;
}

void Handle::ReadTo(void* data, const Allocator::ManageDataPtr& ddata, std::size_t sz) const
{
    MIOPEN_HANDLE_LOCK
    this->Finish();
    auto status = hipMemcpy(data, ddata.get(), sz, hipMemcpyDeviceToHost);
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "Hip error reading from buffer: ");
}

void Handle::Copy(ConstData_t src, Data_t dest, std::size_t size) const
{
    MIOPEN_HANDLE_LOCK
    this->impl->set_ctx();
    auto status = hipMemcpy(dest, src, size, hipMemcpyDeviceToDevice);
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "Hip error copying buffer: ");
}

KernelInvoke Handle::AddKernel(const std::string& algorithm,
                               const std::string& network_config,
                               const std::string& program_name,
                               const std::string& kernel_name,
                               const std::vector<size_t>& vld,
                               const std::vector<size_t>& vgd,
                               const std::string& params,
                               std::size_t cache_index,
                               bool is_kernel_str,
                               const std::string& kernel_src) const
{

    auto obj = this->impl->cache.AddKernel(*this,
                                           algorithm,
                                           network_config,
                                           program_name,
                                           kernel_name,
                                           vld,
                                           vgd,
                                           params,
                                           cache_index,
                                           is_kernel_str,
                                           kernel_src);
    return this->Run(obj);
}

Invoker Handle::PrepareInvoker(const InvokerFactory& factory,
                               const std::vector<solver::KernelInfo>& kernels) const
{
    std::vector<Kernel> built;
    for(auto& k : kernels)
    {
        MIOPEN_LOG_I2("Preparing kernel: " << k.kernel_name);
        const auto kernel = this->impl->cache.AddKernel(*this,
                                                        "",
                                                        "",
                                                        k.kernel_file,
                                                        k.kernel_name,
                                                        k.l_wk,
                                                        k.g_wk,
                                                        k.comp_options,
                                                        kernels.size());
        built.push_back(kernel);
    }
    return factory(built);
}

void Handle::ClearKernels(const std::string& algorithm, const std::string& network_config) const
{
    this->impl->cache.ClearKernels(algorithm, network_config);
}

const std::vector<Kernel>& Handle::GetKernelsImpl(const std::string& algorithm,
                                                  const std::string& network_config) const
{
    return this->impl->cache.GetKernels(algorithm, network_config);
}

bool Handle::HasKernel(const std::string& algorithm, const std::string& network_config) const
{
    return this->impl->cache.HasKernels(algorithm, network_config);
}

KernelInvoke Handle::Run(Kernel k) const
{
    this->impl->set_ctx();
    if(this->impl->enable_profiling || MIOPEN_GPU_SYNC)
        return k.Invoke(this->GetStream(), this->impl->elapsed_time_handler());
    else
        return k.Invoke(this->GetStream());
}

Program Handle::LoadProgram(const std::string& program_name,
                            std::string params,
                            bool is_kernel_str,
                            const std::string& kernel_src) const
{
    this->impl->set_ctx();

    if(!miopen::EndsWith(program_name, ".mlir"))
    {
        params += " -mcpu=" + this->GetTargetProperties().Name();
    }

    auto hsaco = miopen::LoadBinary(this->GetTargetProperties(),
                                    this->GetMaxComputeUnits(),
                                    program_name,
                                    params,
                                    is_kernel_str);
    if(hsaco.empty())
    {
        CompileTimer ct;
        auto p = HIPOCProgram{
            program_name, params, is_kernel_str, this->GetTargetProperties(), kernel_src};
        ct.Log("Kernel", is_kernel_str ? std::string() : program_name);

// Save to cache
#if MIOPEN_ENABLE_SQLITE_KERN_CACHE
        miopen::SaveBinary(p.IsCodeObjectInMemory()
                               ? p.GetCodeObjectBlob()
                               : miopen::LoadFile(p.GetCodeObjectPathname().string()),
                           this->GetTargetProperties(),
                           this->GetMaxComputeUnits(),
                           program_name,
                           params,
                           is_kernel_str);
#else
        auto path = miopen::GetCachePath(false) / boost::filesystem::unique_path();
        if(p.IsCodeObjectInMemory())
            miopen::WriteFile(p.GetCodeObjectBlob(), path);
        else
            boost::filesystem::copy_file(p.GetCodeObjectPathname(), path);
        miopen::SaveBinary(path, this->GetTargetProperties(), program_name, params, is_kernel_str);
#endif
        p.FreeCodeObjectFileStorage();
        return p;
    }
    else
    {
        return HIPOCProgram{program_name, hsaco};
    }
}

bool Handle::HasProgram(const std::string& program_name, const std::string& params) const
{
    return this->impl->cache.HasProgram(program_name, params);
}

void Handle::AddProgram(Program prog,
                        const std::string& program_name,
                        const std::string& params) const
{
    this->impl->cache.AddProgram(prog, program_name, params);
}

void Handle::ClearProgram(const std::string& program_name, const std::string& params) const
{
    this->impl->cache.ClearProgram(program_name, params);
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

void Handle::ResetKernelTime() const { this->impl->profiling_result = 0.0; }
void Handle::AccumKernelTime(float curr_time) const { this->impl->profiling_result += curr_time; }

std::size_t Handle::GetLocalMemorySize() const
{
    int result;
    auto status = hipDeviceGetAttribute(
        &result, hipDeviceAttributeMaxSharedMemoryPerBlock, this->impl->device);
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status);

    return result;
}

std::size_t Handle::GetGlobalMemorySize() const
{
    size_t result;
    auto status = hipDeviceTotalMem(&result, this->impl->device);

    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status);

    return result;
}

std::size_t Handle::GetMaxComputeUnits() const
{
    const std::size_t num_cu = Value(MIOPEN_DEVICE_CU{});
    if(num_cu > 0)
        return num_cu;

    int result;
    auto status =
        hipDeviceGetAttribute(&result, hipDeviceAttributeMultiprocessorCount, this->impl->device);
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status);

    return result;
}

std::size_t Handle::GetImage3dMaxWidth() const
{
    int result;
    auto status = hipDeviceGetAttribute(&result, hipDeviceAttributeMaxGridDimX, this->impl->device);
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status);

    return result;
}

std::size_t Handle::GetWavefrontWidth() const
{
    hipDeviceProp_t props{};
    hipGetDeviceProperties(&props, this->impl->device);
    auto result = static_cast<size_t>(props.warpSize);
    return result;
}

// No HIP API that could return maximum memory allocation size
// for a single object.
std::size_t Handle::GetMaxMemoryAllocSize()
{
    if(m_MaxMemoryAllocSizeCached == 0)
    {
        size_t free, total;
        auto status = hipMemGetInfo(&free, &total);
        if(status != hipSuccess)
            MIOPEN_THROW_HIP_STATUS(status, "Failed getting available memory");
        m_MaxMemoryAllocSizeCached = floor(total * 0.85);
    }

    return m_MaxMemoryAllocSizeCached;
}

std::string Handle::GetDeviceNameImpl() const { return this->impl->get_device_name(); }

std::string Handle::GetDeviceName() const { return this->impl->target_properties.Name(); }

const TargetProperties& Handle::GetTargetProperties() const
{
    return this->impl->target_properties;
}

std::ostream& Handle::Print(std::ostream& os) const
{
    os << "stream: " << this->impl->stream << ", device_id: " << this->impl->device;
    return os;
}

shared<Data_t> Handle::CreateSubBuffer(Data_t data, std::size_t offset, std::size_t) const
{
    auto cdata = reinterpret_cast<char*>(data);
    return {cdata + offset, null_deleter{}};
}

shared<ConstData_t> Handle::CreateSubBuffer(ConstData_t data, std::size_t offset, std::size_t) const
{
    auto cdata = reinterpret_cast<const char*>(data);
    return {cdata + offset, null_deleter{}};
}

#if MIOPEN_USE_ROCBLAS
rocblas_handle_ptr Handle::CreateRocblasHandle() const
{
    rocblas_handle x = nullptr;
    rocblas_create_handle(&x);
    auto result = rocblas_handle_ptr{x};
    rocblas_set_stream(result.get(), GetStream());
    return result;
}
#endif
} // namespace miopen
