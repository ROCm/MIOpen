#include <mlopen/handle.hpp>
#include <mlopen/errors.hpp>
#if MLOPEN_BACKEND_HIPOC
#include <mlopen/kernel_cache.hpp>
#endif
#include <algorithm>

namespace mlopen {

hipDevice_t get_device(int id)
{
    hipDevice_t device;
    auto status = hipDeviceGet(&device, id);
    if (status != hipSuccess) MLOPEN_THROW("No device");
    return device;
}

struct HandleImpl
{
    // typedef MLOPEN_MANAGE_PTR(hipStream_t, hipStreamDestroy) StreamPtr;
    using StreamPtr = std::shared_ptr<typename std::remove_pointer<hipStream_t>::type>;
    using ContextPtr = MLOPEN_MANAGE_PTR(hipCtx_t, hipCtxDestroy);

    HandleImpl()
    {
        this->context = this->create_context();
    }

    ContextPtr create_context()
    {
        hipCtx_t ctx;
        auto status = hipCtxCreate(&ctx, 0, get_device(0));
        ContextPtr result{ctx};
        if (status != hipSuccess) MLOPEN_THROW_HIP_STATUS(status, "Error creating context");
        return result;
    }

    StreamPtr create_stream()
    {
        hipStream_t result;
        auto status = hipStreamCreate(&result);
        if (status != hipSuccess) MLOPEN_THROW_HIP_STATUS(status, "Failed to allocate stream");
        return StreamPtr{result, &hipStreamDestroy};
    }

    static StreamPtr reference_stream(hipStream_t s)
    {
        return StreamPtr{s, null_deleter{}};
    }

    ContextPtr context;
    bool enable_profiling = false;
    std::vector<StreamPtr> streams;
#if MLOPEN_BACKEND_HIPOC
    KernelCache cache;
#endif
};

Handle::Handle (int numStreams, mlopenAcceleratorQueue_t *streams) 
: impl(new HandleImpl())
{
    std::transform(streams, streams+numStreams, std::back_inserter(this->impl->streams), [](hipStream_t x) {
        return HandleImpl::reference_stream(x); 
    });
}

Handle::Handle () 
: impl(new HandleImpl())
{
    this->impl->streams.push_back(impl->create_stream());
}

Handle::~Handle() {}

mlopenAcceleratorQueue_t Handle::GetStream() const
{
    return impl->streams.front().get();
}

void Handle::EnableProfiling(bool enable)
{
    this->impl->enable_profiling = enable;
}

float Handle::GetKernelTime() const
{
    // TODO: Temporary hack until the kernels are added
    if (this->impl->enable_profiling) return 1.0;
    else return 0.0;
}

ManageDataPtr Handle::Create(int sz)
{
    void * result;
    auto status = hipMalloc(&result, sz);
    if (status != hipSuccess) MLOPEN_THROW_HIP_STATUS(status, "Hip error creating buffer: ");
    return ManageDataPtr{result};
}
ManageDataPtr& Handle::WriteTo(const void* data, ManageDataPtr& ddata, int sz)
{
    auto status = hipMemcpy(ddata.get(), data, sz, hipMemcpyHostToDevice);
    if (status != hipSuccess) MLOPEN_THROW_HIP_STATUS(status, "Hip error writing to buffer: ");
    return ddata;
}
void Handle::ReadTo(void* data, const ManageDataPtr& ddata, int sz)
{
    auto status = hipMemcpy(data, ddata.get(), sz, hipMemcpyDeviceToHost);
    if (status != hipSuccess) MLOPEN_THROW_HIP_STATUS(status, "Hip error reading from buffer: ");
}

#if MLOPEN_BACKEND_HIPOC
KernelInvoke Handle::GetKernel(
        const std::string& algorithm,
        const std::string& network_config,
        const std::string& program_name,
        const std::string& kernel_name,
        const std::vector<size_t>& vld,
        const std::vector<size_t>& vgd,
        const std::string& params)
{
    return this->impl->cache.GetKernel(*this, 
            algorithm,
            network_config,
            program_name, 
            kernel_name,
            vld,
            vgd,
            params);
}

KernelInvoke Handle::GetKernel(
    const std::string& algorithm,
    const std::string& network_config)
{
    return this->impl->cache.GetKernel(
            algorithm,
            network_config);
}

Program Handle::LoadProgram(const std::string &program_name, std::string params)
{
    return HIPOCProgram{program_name, params};
}

void Handle::Finish() const
{

}
void Handle::Flush() const
{

}

std::size_t Handle::GetLocalMemorySize()
{
    // TODO: Check error codes
    int dev;
    hipGetDevice(&dev);

    int result;
    hipDeviceGetAttribute(&result, hipDeviceAttributeMaxSharedMemoryPerBlock, 1);

    return result;
}

std::size_t Handle::GetMaxComputeUnits()
{
    // TODO: Check error codes
    int dev;
    hipGetDevice(&dev);

    int result;
    hipDeviceGetAttribute(&result, hipDeviceAttributeMultiprocessorCount , 1);

    return result;
}

std::string Handle::GetDeviceName()
{
    // TODO
    return "Fiji";
}
#endif
}

