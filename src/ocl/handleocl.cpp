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

#include <miopen/handle.hpp>

#include <miopen/binary_cache.hpp>
#include <miopen/config.h>
#include <miopen/env.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle_lock.hpp>
#include <miopen/invoker.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/load_file.hpp>
#include <miopen/logger.hpp>
#include <miopen/manage_ptr.hpp>
#include <miopen/ocldeviceinfo.hpp>
#include <miopen/timer.hpp>

#include <miopen/filesystem.hpp>
#include <boost/filesystem/operations.hpp>

#include <string>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace miopen {

void* default_allocator(void* context, size_t sz)
{
    assert(context != nullptr);
    cl_int status = CL_SUCCESS;
    auto result   = clCreateBuffer(
        reinterpret_cast<cl_context>(context), CL_MEM_READ_ONLY, sz, nullptr, &status);
    if(status != CL_SUCCESS)
    {
        MIOPEN_THROW_CL_STATUS(status, "OpenCL error creating buffer: " + std::to_string(sz));
    }
    return result;
}

void default_deallocator(void*, void* mem) { clReleaseMemObject(DataCast(mem)); }

struct HandleImpl
{

    using AqPtr = miopen::manage_ptr<typename std::remove_pointer<miopenAcceleratorQueue_t>::type,
                                     decltype(&clReleaseCommandQueue),
                                     &clReleaseCommandQueue>;
    using ContextPtr = miopen::manage_ptr<typename std::remove_pointer<cl_context>::type,
                                          decltype(&clReleaseContext),
                                          &clReleaseContext>;

    ContextPtr context  = nullptr;
    AqPtr queue         = nullptr;
    cl_device_id device = nullptr; // NOLINT
    Allocator allocator{};
    KernelCache cache;
    bool enable_profiling  = false;
    float profiling_result = 0.0;
    TargetProperties target_properties;

    std::string get_device_name() const
    {
        std::string name = miopen::GetDeviceInfo<CL_DEVICE_NAME>(device);
        MIOPEN_LOG_NQI("Raw device name: " << name);
        return name;
    }

    ContextPtr create_context()
    {
        // TODO(paul): Change errors to CL errors
        cl_uint numPlatforms;
        cl_platform_id platform = nullptr;
        if(clGetPlatformIDs(0, nullptr, &numPlatforms) != CL_SUCCESS)
        {
            MIOPEN_THROW("clGetPlatformIDs failed. " + std::to_string(numPlatforms));
        }
        if(0 < numPlatforms)
        {
            std::vector<cl_platform_id> platforms(numPlatforms);
            if(clGetPlatformIDs(numPlatforms, platforms.data(), nullptr) != CL_SUCCESS)
            {
                MIOPEN_THROW("clGetPlatformIDs failed.2");
            }
            for(cl_uint i = 0; i < numPlatforms; ++i)
            {
                char pbuf[100];

                if(clGetPlatformInfo(
                       platforms[i], CL_PLATFORM_VENDOR, sizeof(pbuf), pbuf, nullptr) != CL_SUCCESS)
                {
                    MIOPEN_THROW("clGetPlatformInfo failed.");
                }

                platform = platforms[i];
                if(strcmp(pbuf, "Advanced Micro Devices, Inc.") == 0)
                {
                    break;
                }
            }
        }

        /////////////////////////////////////////////////////////////////
        // Create an OpenCL context
        /////////////////////////////////////////////////////////////////
        cl_int status                = 0;
        cl_context_properties cps[3] = {
            CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform), 0};
        cl_context_properties* cprops = (nullptr == platform) ? nullptr : cps;
        ContextPtr result{
            clCreateContextFromType(cprops, CL_DEVICE_TYPE_GPU, nullptr, nullptr, &status)};
        if(status != CL_SUCCESS)
        {
            MIOPEN_THROW_CL_STATUS(status, "Error: Creating Handle. (clCreateContextFromType)");
        }
        return result;
    }
    ContextPtr create_context_from_queue() const
    {
        // FIXME: hack for all the queues on the same context
        // do we need anything special to handle multiple GPUs
        cl_context ctx;
        cl_int status = 0;
        status =
            clGetCommandQueueInfo(queue.get(), CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
        if(status != CL_SUCCESS)
        {
            MIOPEN_THROW_CL_STATUS(status,
                                   "Error: Creating Handle. Cannot Initialize Handle from Queue");
        }
        clRetainContext(ctx);
        return ContextPtr{ctx};
    }
    void ResetProfilingResult() { profiling_result = 0.0; }
    void AccumProfilingResult(float curr_res) { profiling_result += curr_res; }

    void SetProfilingResult(cl_event& e)
    {
        if(this->enable_profiling)
        {
            size_t st, end;
            clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof(size_t), &st, nullptr);
            clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof(size_t), &end, nullptr);
            profiling_result = static_cast<float>(end - st) * 1.0e-6; // NOLINT
        }
    }
};

Handle::Handle(miopenAcceleratorQueue_t stream) : impl(new HandleImpl())
{
    clRetainCommandQueue(stream);
    impl->queue   = HandleImpl::AqPtr{stream};
    impl->device  = miopen::GetDevice(impl->queue.get());
    impl->context = impl->create_context_from_queue();

    this->SetAllocator(nullptr, nullptr, nullptr);
    this->impl->target_properties.Init(this);
    MIOPEN_LOG_NQI(*this);
}

static bool PrintOpenCLDeprecateMsg()
{
    MIOPEN_LOG_W("Please note that the OpenCL backend to MIOpen is being deprecated, ");
    MIOPEN_LOG_W(
        "please port your application to the better supported and functional HIP backend. ");
    MIOPEN_LOG_W("If you have any questions, please reach out to the MIOpen developers at ");
    MIOPEN_LOG_W("https://github.com/ROCm/MIOpen");
    return true;
}

Handle::Handle() : impl(new HandleImpl())
{
    /////////////////////////////////////////////////////////////////
    // Create an OpenCL context
    /////////////////////////////////////////////////////////////////
    static const auto run_once = PrintOpenCLDeprecateMsg();
    std::ignore                = run_once;
    impl->context              = impl->create_context();
    /* First, get the size of device list data */
    cl_uint deviceListSize;
    if(clGetContextInfo(impl->context.get(),
                        CL_CONTEXT_NUM_DEVICES,
                        sizeof(cl_uint),
                        &deviceListSize,
                        nullptr) != CL_SUCCESS)
    {
        MIOPEN_THROW("Error: Getting Handle Info (device list size, clGetContextInfo)");
    }

    if(deviceListSize == 0)
    {
        MIOPEN_THROW("Error: No devices found.");
    }

    /////////////////////////////////////////////////////////////////
    // Detect OpenCL devices
    /////////////////////////////////////////////////////////////////
    std::vector<cl_device_id> devices(deviceListSize);

    /* Now, get the device list data */
    if(clGetContextInfo(impl->context.get(),
                        CL_CONTEXT_DEVICES,
                        deviceListSize * sizeof(cl_device_id),
                        devices.data(),
                        nullptr) != CL_SUCCESS)
    {
        MIOPEN_THROW("Error: Getting Handle Info (device list, clGetContextInfo)");
    }

#ifdef _WIN32
    // Just using the first device as default
    impl->device = devices.at(0);
#else
    // Pick device based on process id
    auto pid = ::getpid();
    assert(pid > 0);
    impl->device = devices.at(pid % devices.size());
#endif

    /////////////////////////////////////////////////////////////////
    // Create an OpenCL command queue
    /////////////////////////////////////////////////////////////////
    cl_int status = 0;
#ifdef CL_VERSION_2_0
    const cl_queue_properties cq_props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};

    impl->queue = HandleImpl::AqPtr{
        clCreateCommandQueueWithProperties(impl->context.get(), impl->device, cq_props, &status)};
#else
    impl->queue  = HandleImpl::AqPtr{clCreateCommandQueue(
        impl->context.get(), impl->device, CL_QUEUE_PROFILING_ENABLE, &status)};
#endif
    if(status != CL_SUCCESS)
    {
        MIOPEN_THROW("Error creating command queue");
    }
    this->SetAllocator(nullptr, nullptr, nullptr);
    this->impl->target_properties.Init(this);
    MIOPEN_LOG_NQI(*this);
}

Handle::Handle(Handle&&) noexcept = default;
Handle::~Handle()                 = default;

void Handle::SetStream(miopenAcceleratorQueue_t streamID) const
{
    if(streamID == nullptr)
    {
        MIOPEN_THROW("Error setting stream to nullptr");
    }

    clRetainCommandQueue(streamID);
    impl->queue = HandleImpl::AqPtr{streamID};
    this->impl->target_properties.Init(this);
    MIOPEN_LOG_NQI(*this);
}

void Handle::SetStreamFromPool(int) const { MIOPEN_THROW("Error streamPool unsupported at OCl"); }
void Handle::ReserveExtraStreamsInPool(int) const
{
    MIOPEN_THROW("Error streamPool unsupported at OCl");
}

miopenAcceleratorQueue_t Handle::GetStream() const { return impl->queue.get(); }

void Handle::SetAllocator(miopenAllocatorFunction allocator,
                          miopenDeallocatorFunction deallocator,
                          void* allocatorContext) const
{
    if(allocator == nullptr && allocatorContext != nullptr)
    {
        MIOPEN_THROW("Allocator context can not be used with the default allocator");
    }
    this->impl->allocator.allocator   = allocator == nullptr ? default_allocator : allocator;
    this->impl->allocator.deallocator = deallocator == nullptr ? default_deallocator : deallocator;

    this->impl->allocator.context =
        allocatorContext == nullptr ? this->impl->context.get() : allocatorContext;
}

void Handle::EnableProfiling(bool enable) const { this->impl->enable_profiling = enable; }

void Handle::ResetKernelTime() const { this->impl->ResetProfilingResult(); }
void Handle::AccumKernelTime(float curr_time) const { this->impl->AccumProfilingResult(curr_time); }

float Handle::GetKernelTime() const { return this->impl->profiling_result; }

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
                               const std::vector<solver::KernelInfo>& kernels,
                               std::vector<Program>* programs_out) const
{
    std::ignore = programs_out;

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

std::vector<Kernel> Handle::GetKernelsImpl(const std::string& algorithm,
                                           const std::string& network_config) const
{
    return this->impl->cache.GetKernels(algorithm, network_config);
}

KernelInvoke Handle::Run(Kernel k, bool coop_launch) const
{
    if(coop_launch)
        MIOPEN_THROW(miopenStatusInternalError);

    auto q = this->GetStream();
    if(this->impl->enable_profiling || MIOPEN_GPU_SYNC)
    {
        return k.Invoke(q,
                        std::bind(&HandleImpl::SetProfilingResult,
                                  std::ref(*this->impl),
                                  std::placeholders::_1));
    }
    else
    {
        return k.Invoke(q);
    }
}

Program Handle::LoadProgram(const std::string& program_name,
                            std::string params,
                            const std::string& kernel_src,
                            bool force_attach_binary) const
{
    // Binary serialization is not supported on OpenCL anyway
    std::ignore = force_attach_binary;

    auto hsaco = miopen::LoadBinary(
        this->GetTargetProperties(), this->GetMaxComputeUnits(), program_name, params);
    if(hsaco.empty())
    {
        CompileTimer ct;
        auto p = miopen::LoadProgram(miopen::GetContext(this->GetStream()),
                                     miopen::GetDevice(this->GetStream()),
                                     this->GetTargetProperties(),
                                     program_name,
                                     params,
                                     kernel_src);
        ct.Log("Kernel", program_name);

// Save to cache
#if MIOPEN_ENABLE_SQLITE_KERN_CACHE
        std::string binary;
        miopen::GetProgramBinary(p, binary);
        miopen::SaveBinary(
            binary, this->GetTargetProperties(), this->GetMaxComputeUnits(), program_name, params);
#else
        auto path = miopen::GetCachePath(false) / boost::filesystem::unique_path().string();
        miopen::SaveProgramBinary(p, path.string());
        miopen::SaveBinary(path, this->GetTargetProperties(), program_name, params);
#endif
        return p;
    }
    else
    {
        return LoadBinaryProgram(miopen::GetContext(this->GetStream()),
                                 miopen::GetDevice(this->GetStream()),
#if MIOPEN_ENABLE_SQLITE_KERN_CACHE
                                 hsaco);
#else
                                 miopen::LoadFile(hsaco));
#endif
    }
}

void Handle::ClearProgram(const std::string& program_name, const std::string& params) const
{
    this->impl->cache.ClearProgram(program_name, params);
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

void Handle::Finish() const { clFinish(this->GetStream()); }

void Handle::Flush() const { clFlush(this->GetStream()); }

bool Handle::IsProfilingEnabled() const { return this->impl->enable_profiling; }

std::size_t Handle::GetLocalMemorySize() const
{
    return miopen::GetDeviceInfo<CL_DEVICE_LOCAL_MEM_SIZE>(miopen::GetDevice(this->GetStream()));
}

std::size_t Handle::GetGlobalMemorySize() const
{
    return miopen::GetDeviceInfo<CL_DEVICE_GLOBAL_MEM_SIZE>(miopen::GetDevice(this->GetStream()));
}

std::string Handle::GetDeviceNameImpl() const { return this->impl->get_device_name(); }

std::string Handle::GetDeviceName() const { return this->impl->target_properties.Name(); }

const TargetProperties& Handle::GetTargetProperties() const
{
    return this->impl->target_properties;
}

std::ostream& Handle::Print(std::ostream& os) const
{
    os << "stream: " << this->impl->queue.get() << ", device_id: " << this->impl->device;
    return os;
}

std::size_t Handle::GetMaxMemoryAllocSize()
{
    if(m_MaxMemoryAllocSizeCached == 0)
        m_MaxMemoryAllocSizeCached = miopen::GetDeviceInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>(
            miopen::GetDevice(this->GetStream()));
    return m_MaxMemoryAllocSizeCached;
}

bool Handle::CooperativeLaunchSupported() const { return false; }

std::size_t Handle::GetMaxComputeUnits() const
{
    return miopen::GetDeviceInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(miopen::GetDevice(this->GetStream()));
}

std::size_t Handle::GetImage3dMaxWidth() const
{
    return miopen::GetDeviceInfo<CL_DEVICE_IMAGE3D_MAX_WIDTH>(miopen::GetDevice(this->GetStream()));
}

std::size_t Handle::GetWavefrontWidth() const
{
    return miopen::GetDeviceInfo<CL_DEVICE_WAVEFRONT_WIDTH_AMD>(
        miopen::GetDevice(this->GetStream()));
}

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
    cl_int status = clEnqueueWriteBuffer(
        this->GetStream(), ddata.get(), CL_TRUE, 0, sz, data, 0, nullptr, nullptr);
    if(status != CL_SUCCESS)
    {
        MIOPEN_THROW_CL_STATUS(status, "OpenCL error writing to buffer: " + std::to_string(sz));
    }
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
    auto status =
        clEnqueueReadBuffer(this->GetStream(), ddata, CL_TRUE, 0, sz, data, 0, nullptr, nullptr);
    if(status != CL_SUCCESS)
    {
        MIOPEN_THROW_CL_STATUS(status, "OpenCL error reading from buffer: " + std::to_string(sz));
    }
}

void Handle::Copy(ConstData_t src, Data_t dest, std::size_t size) const
{
    MIOPEN_HANDLE_LOCK
    this->Finish();
    auto status =
        clEnqueueCopyBuffer(this->GetStream(), src, dest, 0, 0, size, 0, nullptr, nullptr);
    if(status != CL_SUCCESS)
    {
        MIOPEN_THROW_CL_STATUS(status, "OpenCL error copying buffer: " + std::to_string(size));
    }
}

shared<Data_t> Handle::CreateSubBuffer(Data_t data, std::size_t offset, std::size_t size) const
{
    MIOPEN_HANDLE_LOCK
    struct region
    {
        std::size_t origin;
        std::size_t size;
    };
    cl_int error = 0;
    auto r       = region{offset, size};
    auto mem = clCreateSubBuffer(data, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &r, &error);

    if(error != CL_SUCCESS)
    {
        auto ss = std::ostringstream{};
        ss << "Failed to allocate a subbuffer from 0x" << std::hex
           << reinterpret_cast<std::ptrdiff_t>(data) << " with an offset 0x" << r.origin
           << " and size 0x" << r.size << ". ";

        if(error == CL_MISALIGNED_SUB_BUFFER_OFFSET)
        {
            ss << "CL_DEVICE_MEM_BASE_ADDR_ALIGN: 0x" << std::hex
               << GetDeviceInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>(GetDevice(this->GetStream()))
               << " bits. ";
        }

        ss << "OpenCL error:";
        MIOPEN_THROW_CL_STATUS(error, ss.str());
    }

    return {mem, manage_deleter<decltype(&clReleaseMemObject), &clReleaseMemObject>{}};
}

} // namespace miopen
