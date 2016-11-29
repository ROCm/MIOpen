#include <mlopen/handle.hpp>
#include <mlopen/errors.hpp>
#if MLOPEN_BACKEND_HIPOC
#include <mlopen/kernel_cache.hpp>
#endif
#include <algorithm>

namespace mlopen {

struct HandleImpl
{
    // typedef MLOPEN_MANAGE_PTR(hipStream_t, hipStreamDestroy) StreamPtr;
    typedef std::shared_ptr<typename std::remove_pointer<hipStream_t>::type> StreamPtr;

    StreamPtr create_stream()
    {
        hipStream_t result;
        auto status = hipStreamCreate(&result);
        if (status != hipSuccess) MLOPEN_THROW("Failed to allocate stream");
        return StreamPtr{result, &hipStreamDestroy};
    }

    static StreamPtr reference_stream(hipStream_t s)
    {
        return StreamPtr{s, null_deleter{}};
    }

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
    if (status != hipSuccess) MLOPEN_THROW("Hip error creating buffer: " + std::to_string(status));
    return ManageDataPtr{result};
}
ManageDataPtr& Handle::WriteTo(const void* data, ManageDataPtr& ddata, int sz)
{
    auto status = hipMemcpy(ddata.get(), data, sz, hipMemcpyHostToDevice);
    if (status != hipSuccess) MLOPEN_THROW("Hip error writing to buffer: " + std::to_string(status));
    return ddata;
}
void Handle::ReadTo(void* data, const ManageDataPtr& ddata, int sz)
{
    auto status = hipMemcpy(data, ddata.get(), sz, hipMemcpyDeviceToHost);
    if (status != hipSuccess) MLOPEN_THROW("Hip error reading from buffer: " + std::to_string(status));
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
    Device_id dev;
    hipGetDevice(&dev);

    int result;
    hipDeviceGetAttribute(&result, hipDeviceAttributeMaxSharedMemoryPerBlock, 1);

    return result;
}

std::string Handle::GetDeviceName()
{
    // TODO
    return "Fiji";
}
#endif
}

