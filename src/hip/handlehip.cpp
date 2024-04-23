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
#include <miopen/handle_lock.hpp>
#include <miopen/invoker.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/logger.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/timer.hpp>

#if !MIOPEN_ENABLE_SQLITE_KERN_CACHE
#include <miopen/write_file.hpp>
#include <boost/filesystem/operations.hpp>
#endif

#include <miopen/filesystem.hpp>
#include <miopen/load_file.hpp>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <thread>
#include <mutex>
#include <shared_mutex>

/// hipMemGetInfo constantly fails on gfx906/900 and Navi21.
/// Brute-force W/A: return fixed values.
#define WORKAROUND_FAULTY_HIPMEMGETINFO_VEGA_NAVI2X (HIP_PACKAGE_VERSION_FLAT >= 5007000000ULL)

MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_DEVICE_CU)

namespace miopen {

namespace {

hipError_t hip_mem_get_info_wrapper(std::size_t* const free, std::size_t* const total)
{
#if WORKAROUND_FAULTY_HIPMEMGETINFO_VEGA_NAVI2X
    const auto status = hipMemGetInfo(free, total);
    if(status == hipSuccess)
        return status;
    MIOPEN_LOG_W("hipMemGetInfo error, status: " << status);
    assert(free != nullptr && total != nullptr);
    *free  = 16ULL * 1024 * 1024 * 1024; // 16 GiB
    *total = *free;
    return hipSuccess;
#else
    return hipMemGetInfo(free, total);
#endif
}

std::size_t GetAvailableMemory()
{
    size_t free, total;
    auto status = hip_mem_get_info_wrapper(&free, &total);
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "Failed getting available memory");
    return free;
}

void* default_allocator(void*, size_t sz)
{
    const auto available = GetAvailableMemory();
    MIOPEN_LOG_I2("GetAvailableMemory " << available);
    if(sz > available)
        MIOPEN_LOG_I("GetAvailableMemory reports unsufficient memory to allocate " << sz);
    void* ptr;
    const auto status = hipMalloc(&ptr, sz);
    if(status == hipSuccess)
    {
        MIOPEN_LOG_I2("hipMalloc " << sz << " at " << ptr << " Ok");
        return ptr;
    }
    const auto status_host = hipHostMalloc(&ptr, sz);
    if(status_host == hipSuccess)
    {
        MIOPEN_LOG_I2("hipHostMalloc " << sz << " at " << ptr << " Ok");
        return ptr;
    }
    MIOPEN_LOG_W("hipMalloc " << sz << " status: " << status);
    MIOPEN_THROW_HIP_STATUS(status_host, "hipHostMalloc " + std::to_string(sz));
}

[[maybe_unused]] inline std::string to_string(void* const ptr)
{
    std::ostringstream oss;
    oss << ptr;
    return oss.str();
}

void default_deallocator(void*, void* mem)
{
    size_t size = 0;
    auto status = hipMemPtrGetInfo(mem, &size);
    if(status != hipSuccess)
        MIOPEN_LOG_W("hipMemPtrGetInfo at " << mem << " status: " << status);
    status = hipFree(mem);
    if(status != hipSuccess)
    {
        MIOPEN_THROW_HIP_STATUS(status,
                                "hipFree " + std::to_string(size) + " at " + to_string(mem));
    }
    else
        MIOPEN_LOG_I2("hipFree " << size << " at " << mem << " Ok");
}

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

// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
static thread_local unsigned int meopenHandle_current_stream_id = 0;
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

    StreamPtr create_stream_non_blocking()
    {
        hipStream_t result;
        auto status = hipStreamCreateWithFlags(&result, hipStreamNonBlocking);
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
        const std::string name(props.gcnArchName);
        MIOPEN_LOG_NQI("Raw device name: " << name);
        return name; // NOLINT (performance-no-automatic-move)
    }

    std::shared_timed_mutex stream_pool_mutex;
    // the main stream and main rocblas_handle rhandle_

#if MIOPEN_USE_ROCBLAS
    rocblas_handle_ptr rhandle_;
    using RocblasHandlePtrPool = std::vector<rocblas_handle_ptr>;
#endif

    StreamPtr root_stream = nullptr;

    using StreamPtrPool = std::vector<StreamPtr>;
    struct MultiStreamResourses
    {

#if MIOPEN_USE_ROCBLAS
        //  (rocBLAS doc):rocBLAS handle contains allocated device memory that must not be shared by
        //  multiple
        //  asynchronous streams at the same time.
        //  Each parallel thread must use its own rocblas_handle.
        RocblasHandlePtrPool rhandle_pool;
        void add_resours(StreamPtr s_ptr, rocblas_handle_ptr r_ptr)
        {
            stream_pool.push_back(std::move(s_ptr));
            rhandle_pool.push_back(std::move(r_ptr));
        }
#else
        void add_stream(StreamPtr s_ptr) { stream_pool.push_back(s_ptr); }
#endif
        //  stream_pool used as cache for parallel streams created by MIOpen.
        StreamPtrPool stream_pool;
    };

    MultiStreamResourses* ms_resourse_ptr;
    std::map<miopenAcceleratorQueue_t, MultiStreamResourses> extra_stream_map;

    bool enable_profiling  = false;
    float profiling_result = 0.0;
    int device             = -1;
    Allocator allocator{};
    KernelCache cache;
    TargetProperties target_properties;
};

Handle::Handle(miopenAcceleratorQueue_t stream) : impl(std::make_unique<HandleImpl>())
{
    meopenHandle_current_stream_id = 0;
    this->impl->device             = get_device_id();

    if(stream == nullptr)
        this->impl->root_stream = HandleImpl::reference_stream(nullptr);
    else
        this->impl->root_stream = HandleImpl::reference_stream(stream);

    this->impl->extra_stream_map.emplace(stream, HandleImpl::MultiStreamResourses());

    this->impl->ms_resourse_ptr = &(this->impl->extra_stream_map.begin()->second);

    this->SetAllocator(nullptr, nullptr, nullptr);

#if MIOPEN_USE_ROCBLAS
    this->impl->rhandle_ = CreateRocblasHandle(stream);
#endif
    this->impl->target_properties.Init(this);
    MIOPEN_LOG_NQI(*this);
}

Handle::Handle() : impl(std::make_unique<HandleImpl>())
{
    meopenHandle_current_stream_id = 0;
#if MIOPEN_BUILD_DEV
    this->impl->device      = set_default_device();
    this->impl->root_stream = impl->create_stream();
#else
    this->impl->device      = get_device_id();
    this->impl->root_stream = HandleImpl::reference_stream(nullptr);
#endif
    auto root_stream = this->impl->root_stream.get();
    this->impl->extra_stream_map.emplace(root_stream, HandleImpl::MultiStreamResourses());

    this->impl->ms_resourse_ptr = &(this->impl->extra_stream_map.begin()->second);

    this->SetAllocator(nullptr, nullptr, nullptr);

#if MIOPEN_USE_ROCBLAS
    this->impl->rhandle_ = CreateRocblasHandle(root_stream);
#endif
    this->impl->target_properties.Init(this);
    MIOPEN_LOG_NQI(*this);
}

Handle::~Handle() {}

// not MT safe
void Handle::SetStream(miopenAcceleratorQueue_t streamID) const
{
    meopenHandle_current_stream_id = 0;

    this->impl->root_stream = HandleImpl::reference_stream(streamID);

    this->impl->extra_stream_map.emplace(streamID, HandleImpl::MultiStreamResourses());

    this->impl->ms_resourse_ptr = &(this->impl->extra_stream_map[streamID]);

#if MIOPEN_USE_ROCBLAS
    rocblas_set_stream(this->rhandle().get(), this->GetStream());
#endif
    this->impl->target_properties.Init(this);
    MIOPEN_LOG_NQI(*this);
}

void Handle::SetStreamFromPool(int streamID) const { meopenHandle_current_stream_id = streamID; }

void Handle::ReserveExtraStreamsInPool(int cnt) const
{
    std::unique_lock<std::shared_timed_mutex> lock(this->impl->stream_pool_mutex);

    int last_stream_id = this->impl->ms_resourse_ptr->stream_pool.size();

    if(last_stream_id < cnt)
    {
        for(; last_stream_id < cnt; last_stream_id++)
        {
            auto new_stream = this->impl->create_stream_non_blocking();
#if MIOPEN_USE_ROCBLAS
            auto new_rhandle = CreateRocblasHandle(new_stream.get());
            this->impl->ms_resourse_ptr->add_resours(std::move(new_stream), std::move(new_rhandle));
#else
            this->impl->ms_resourse_ptr->add_stream(std::move(new_stream));
#endif
        }
    }
}

miopenAcceleratorQueue_t Handle::GetStream() const
{
    if(meopenHandle_current_stream_id == 0)
        return impl->root_stream.get();
    // locking only if handle in multistream mode
    std::shared_lock<std::shared_timed_mutex> lock(this->impl->stream_pool_mutex);
    return this->impl->ms_resourse_ptr->stream_pool.at(meopenHandle_current_stream_id - 1).get();
}

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
    ReadTo(data, ddata.get(), sz);
}

void Handle::ReadTo(void* data, ConstData_t ddata, std::size_t sz) const
{
    MIOPEN_HANDLE_LOCK
    this->Finish();
    auto status = hipMemcpy(data, ddata, sz, hipMemcpyDeviceToHost);
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
                            const std::string& kernel_src) const
{
    this->impl->set_ctx();
    std::string arch_name = this->GetTargetProperties().Name();

    std::string orig_params = params; // make a copy for target ID fallback

    if(!miopen::EndsWith(program_name, ".mlir"))
        params = params + " -mcpu=" + this->GetTargetProperties().Name();

    auto hsaco = miopen::LoadBinary(
        this->GetTargetProperties(), this->GetMaxComputeUnits(), program_name, params);
    if(hsaco.empty())
    {
        const auto arch_target_id = miopen::SplitDelim(arch_name, ':');
        if(arch_target_id.size() > 1)
        {
            // The target name has target ID in there, fall back on the generic code object
            const auto base_arch = arch_target_id.at(0);
            hsaco                = miopen::LoadBinary(this->GetTargetProperties(),
                                       this->GetMaxComputeUnits(),
                                       program_name,
                                       orig_params + " -mcpu=" + base_arch);
        }
    }

    // Still unable to find the object, build it with the available compiler possibly a target ID
    // specific code object
    if(hsaco.empty())
    {
        CompileTimer ct;
        auto p = HIPOCProgram{program_name, params, this->GetTargetProperties(), kernel_src};
        ct.Log("Kernel", program_name);

// Save to cache
#if MIOPEN_ENABLE_SQLITE_KERN_CACHE
        miopen::SaveBinary(p.IsCodeObjectInMemory() ? p.GetCodeObjectBlob()
                                                    : miopen::LoadFile(p.GetCodeObjectPathname()),
                           this->GetTargetProperties(),
                           this->GetMaxComputeUnits(),
                           program_name,
                           params);
#else
        auto path = miopen::GetCachePath(false) / boost::filesystem::unique_path().string();
        if(p.IsCodeObjectInMemory())
            miopen::WriteFile(p.GetCodeObjectBlob(), path);
        else
            fs::copy_file(p.GetCodeObjectPathname(), path);
        miopen::SaveBinary(path, this->GetTargetProperties(), program_name, params);
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
    const std::size_t num_cu = Value(ENV(MIOPEN_DEVICE_CU));
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
        auto status = hip_mem_get_info_wrapper(&free, &total);
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
    os << "stream: " << GetStream() << ", device_id: " << this->impl->device;
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

const rocblas_handle_ptr& Handle::rhandle() const
{
    if(meopenHandle_current_stream_id == 0)
        return this->impl->rhandle_;
    // locking only if handle in multistream mode
    std::shared_lock<std::shared_timed_mutex> lock(this->impl->stream_pool_mutex);
    return this->impl->ms_resourse_ptr->rhandle_pool.at(meopenHandle_current_stream_id - 1);
}

rocblas_handle_ptr Handle::CreateRocblasHandle(miopenAcceleratorQueue_t stream) const
{
    rocblas_handle x = nullptr;
    rocblas_create_handle(&x);
    auto result = rocblas_handle_ptr{x};
    rocblas_set_stream(result.get(), stream);
    return result;
}
#endif
} // namespace miopen
