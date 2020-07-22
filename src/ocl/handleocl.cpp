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
#include <miopen/device_name.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle_lock.hpp>
#include <miopen/invoker.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/load_file.hpp>
#include <miopen/logger.hpp>
#include <miopen/manage_ptr.hpp>
#include <miopen/ocldeviceinfo.hpp>
#include <miopen/timer.hpp>

#if MIOPEN_USE_MIOPENGEMM
#include <miopen/gemm_geometry.hpp>
#endif

#include <boost/filesystem.hpp>

#include <string>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace miopen {

#ifndef NDEBUG
void dumpKernel(cl_kernel kern,
                const std::string& kernel_name,
                const std::vector<size_t>& vld,
                const std::vector<size_t>& vgd,
                const std::string& params)
{
    static int dumpOpenCLFileCounter = 0;
    static std::vector<cl_kernel> kernList;
    for(auto it = kernList.begin(); it != kernList.end(); it++)
        if(*it == kern)
            return;
    kernList.push_back(kern);
    std::string work;
    for(size_t i = 0; i < vgd.size(); i++)
    {
        if(i)
            work += ",";
        work += std::to_string(vgd[i]);
    }
    for(size_t i = 0; i < vld.size(); i++)
    {
        work += i ? "," : "/";
        work += std::to_string(vld[i]);
    }
    auto getValueFromParams = [&](const std::string& par, int& value, const char* define) {
        const char* q = strstr(par.c_str(), define);
        if(q)
            value = atoi(q + strlen(define));
    };
    int an = 0, ac = 0, ah = 0, aw = 0, ax = 0, ay = 0, ak = 0, ap = 0, aq = 0, au = 1, av = 1,
        aP = 0, aQ = 0, af = 1;
    getValueFromParams(params, an, "-D MLO_BATCH_SZ=");
    getValueFromParams(params, ac, "-D MLO_N_INPUTS=");
    getValueFromParams(params, ac, "-D MLO_N_IN_CHNLS=");
    getValueFromParams(params, ah, "-D MLO_IN_HEIGHT=");
    getValueFromParams(params, aw, "-D MLO_IN_WIDTH=");
    getValueFromParams(params, ak, "-D MLO_N_OUTPUTS=");
    getValueFromParams(params, ak, "-D MLO_N_OUT_CHNLS=");
    getValueFromParams(params, aP, "-D MLO_OUT_HEIGHT=");
    getValueFromParams(params, aQ, "-D MLO_OUT_WIDTH=");
    getValueFromParams(params, ay, "-D MLO_FILTER_SIZE1=");
    getValueFromParams(params, ax, "-D MLO_FILTER_SIZE0=");
    getValueFromParams(params, ap, "-D MLO_FILTER_PAD1=");
    getValueFromParams(params, aq, "-D MLO_FILTER_PAD0=");
    getValueFromParams(params, av, "-D MLO_FILTER_STRIDE1=");
    getValueFromParams(params, au, "-D MLO_FILTER_STRIDE0=");
    getValueFromParams(params, ay, "-D MLO_FLTR_SZ1=");
    getValueFromParams(params, ax, "-D MLO_FLTR_SZ0=");
    getValueFromParams(params, ap, "-D MLO_FLTR_PAD_SZ1=");
    getValueFromParams(params, aq, "-D MLO_FLTR_PAD_SZ0=");
    getValueFromParams(params, av, "-D MLO_FLTR_STRIDE1=");
    getValueFromParams(params, au, "-D MLO_FLTR_STRIDE0=");
    getValueFromParams(params, af, "-D MLO_DIR_FORWARD=");
    int isize = an * ac * ah * aw * 4;
    int osize = an * ak * aP * aQ * 4;
    int wsize = ak * ac * ay * ax * 4;
    if(!isize || !osize || !wsize)
    {
        if(params.size() > 0)
            printf("dumpKernel: can't dump kernel %s missing macros in params: %s\n",
                   kernel_name.c_str(),
                   params.c_str());
        return;
    }
    dumpOpenCLFileCounter++;
    cl_program prog = nullptr;
    clGetKernelInfo(kern, CL_KERNEL_PROGRAM, sizeof(prog), &prog, nullptr);
    cl_uint num_arg = 0;
    clGetKernelInfo(kern, CL_KERNEL_NUM_ARGS, sizeof(num_arg), &num_arg, nullptr);
    size_t sizeK = 0;
    clGetProgramInfo(prog, CL_PROGRAM_SOURCE, 0, nullptr, &sizeK);
    std::vector<char> bufK(sizeK + 1);
    char* buf   = bufK.data();
    size_t size = 0;
    clGetProgramInfo(prog, CL_PROGRAM_SOURCE, sizeK, buf, &size);
    buf[size] = 0;
    char fileName[1024];
    FILE* fp;
    sprintf(fileName, "dump_%03d_command.txt", dumpOpenCLFileCounter);
    fp = fopen(fileName, "w");
    if(!fp)
    {
        printf("ERROR: unable to create: %s\n", fileName);
    }
    else
    {
        if(af)
        {
            fprintf(fp,
                    "execkern -bo -cl-std=CL2.0 dump_%03d_kernel.cl -k %s if#%d:dump_fwd_in.bin "
                    "if#%d:dump_fwd_wei.bin of#%d:#intmp.bin#/+1e%d/dump_fwd_out_cpu.bin %s %s -- "
                    "comment -n %d -c %d -H %d -W %d -x %d -y %d -k %d -p %d -q %d -u %d -v %d -- "
                    "P %d Q %d",
                    dumpOpenCLFileCounter,
                    kernel_name.c_str(),
                    isize,
                    wsize,
                    osize,
                    -6,
                    num_arg > 3 ? "iv#0 " : "",
                    work.c_str(),
                    an,
                    ac,
                    ah,
                    aw,
                    ax,
                    ay,
                    ak,
                    ap,
                    aq,
                    au,
                    av,
                    aP,
                    aQ);
        }
        else
        {
            fprintf(fp,
                    "execkern -bo -cl-std=CL2.0 dump_%03d_kernel.cl -k %s if#%d:dump_bwd_out.bin "
                    "if#%d:dump_bwd_wei.bin of#%d:#outtmp.bin#/+1e%d/dump_bwd_in_cpu.bin %s %s -- "
                    "comment -n %d -c %d -H %d -W %d -x %d -y %d -k %d -p %d -q %d -u %d -v %d -- "
                    "P %d Q %d",
                    dumpOpenCLFileCounter,
                    kernel_name.c_str(),
                    isize,
                    wsize,
                    osize,
                    -9,
                    num_arg > 3 ? "iv#0 " : "",
                    work.c_str(),
                    an,
                    ac,
                    ah,
                    aw,
                    ax,
                    ay,
                    ak,
                    ap,
                    aq,
                    au,
                    av,
                    aP,
                    aQ);
        }
        fclose(fp);
        printf("*** OpenCL kernel %s command dumped into %s with work %s\n",
               kernel_name.c_str(),
               fileName,
               work.c_str());
    }
    sprintf(fileName, "dump_%03d_kernel.cl", dumpOpenCLFileCounter);
    fp = fopen(fileName, "w");
    if(!fp)
    {
        printf("ERROR: unable to create: %s\n", fileName);
    }
    else
    {
        const char* s = params.c_str();
        fprintf(fp, "//[compiler-options] %s\n", s);
        for(const char* t = s; (t = strstr(t, "-D")) != nullptr;)
        {
            t += 2;
            while(*t && (*t == ' ' || *t == '\t'))
                t++;
            fprintf(fp, "#define ");
            while(*t && *t != ' ' && *t != '\t' && *t != '=')
                fprintf(fp, "%c", *t++);
            if(*t == '=')
            {
                fprintf(fp, " ");
                t++;
                while(*t && *t != ' ' && *t != '\t')
                    fprintf(fp, "%c", *t++);
            }
            fprintf(fp, "\n");
        }
        for(const char* p = buf; *p; p++)
            if(*p != '\r')
                fprintf(fp, "%c", *p);
        fclose(fp);
        printf("*** OpenCL kernel %s source dumped into %s with work %s\n",
               kernel_name.c_str(),
               fileName,
               work.c_str());
    }
}
#endif

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
    impl->context = impl->create_context_from_queue();

    this->SetAllocator(nullptr, nullptr, nullptr);
}

Handle::Handle() : impl(new HandleImpl())
{
    /////////////////////////////////////////////////////////////////
    // Create an OpenCL context
    /////////////////////////////////////////////////////////////////

    impl->context = impl->create_context();
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
    impl->device  = devices.at(pid % devices.size());
#endif

#if !MIOPEN_INSTALLABLE
    // TODO: Store device name in handle
    std::string deviceName = miopen::GetDeviceInfo<CL_DEVICE_NAME>(impl->device);
    ParseDevName(deviceName);
    MIOPEN_LOG_NQI("Device name: " << deviceName);
#endif

    /////////////////////////////////////////////////////////////////
    // Create an OpenCL command queue
    /////////////////////////////////////////////////////////////////
    cl_int status = 0;
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
    impl->queue = HandleImpl::AqPtr{clCreateCommandQueue(
        impl->context.get(), impl->device, CL_QUEUE_PROFILING_ENABLE, &status)};
#ifdef __clang__
#pragma clang diagnostic pop
#endif
    if(status != CL_SUCCESS)
    {
        MIOPEN_THROW("Creating Command Queue. (clCreateCommandQueue)");
    }
    this->SetAllocator(nullptr, nullptr, nullptr);
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

bool Handle::HasKernel(const std::string& algorithm, const std::string& network_config) const
{
    return this->impl->cache.HasKernels(algorithm, network_config);
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
                            bool is_kernel_str,
                            const std::string& kernel_src) const
{
    auto hsaco = miopen::LoadBinary(
        this->GetDeviceName(), this->GetMaxComputeUnits(), program_name, params, is_kernel_str);
    if(hsaco.empty())
    {
        CompileTimer ct;
        auto p = miopen::LoadProgram(miopen::GetContext(this->GetStream()),
                                     miopen::GetDevice(this->GetStream()),
                                     program_name,
                                     params,
                                     is_kernel_str,
                                     kernel_src);
        ct.Log("Kernel", program_name);

// Save to cache
#if MIOPEN_ENABLE_SQLITE_KERN_CACHE
        std::string binary;
        miopen::GetProgramBinary(p, binary);
        miopen::SaveBinary(binary,
                           this->GetDeviceName(),
                           this->GetMaxComputeUnits(),
                           program_name,
                           params,
                           is_kernel_str);
#else
        auto path = miopen::GetCachePath(false) / boost::filesystem::unique_path();
        miopen::SaveProgramBinary(p, path.string());
        miopen::SaveBinary(
            path.string(), this->GetDeviceName(), program_name, params, is_kernel_str);
#endif
        return std::move(p);
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

std::string Handle::GetDeviceName() const
{
    std::string name = miopen::GetDeviceInfo<CL_DEVICE_NAME>(miopen::GetDevice(this->GetStream()));
    ParseDevName(name);
    return GetDeviceNameFromMap(name);
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
    MIOPEN_HANDLE_LOCK
    this->Finish();
    auto status = clEnqueueReadBuffer(
        this->GetStream(), ddata.get(), CL_TRUE, 0, sz, data, 0, nullptr, nullptr);
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

shared<Data_t> Handle::CreateSubBuffer(Data_t data, std::size_t offset, std::size_t size)
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
    return {mem, manage_deleter<decltype(&clReleaseMemObject), &clReleaseMemObject>{}};
}

} // namespace miopen
