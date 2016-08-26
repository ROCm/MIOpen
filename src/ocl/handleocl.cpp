#include <mlopen/handle.hpp>
#include <mlopen/manage_ptr.hpp>
#include <mlopen/errors.hpp>
#include <mlopen/kernel_cache.hpp>
#include <string>

namespace mlopen {

struct HandleImpl
{

    typedef MLOPEN_MANAGE_PTR(mlopenAcceleratorQueue_t, clReleaseCommandQueue) AqPtr;
    typedef MLOPEN_MANAGE_PTR(cl_context, clReleaseContext) ContextPtr;

    ContextPtr context;
    std::vector<AqPtr> queues;
    KernelCache cache;
    bool enable_profiling;
    float profiling_result;

    ContextPtr create_context()
    {
        // TODO: Change errors to CL errors
        cl_uint numPlatforms;
        cl_platform_id platform = nullptr;
        if(clGetPlatformIDs(0, nullptr, &numPlatforms) != CL_SUCCESS)
        {
            fprintf(stderr,"clGetPlatformIDs failed. %u",numPlatforms);
            throw mlopenStatusInternalError;
        }
        if (0 < numPlatforms) 
        {
            std::vector<cl_platform_id> platforms(numPlatforms);
            if(clGetPlatformIDs(numPlatforms, platforms.data(), nullptr) != CL_SUCCESS)
            {
                perror( "clGetPlatformIDs failed.2");
                throw mlopenStatusInternalError;
            }
            for (int i = 0; i < numPlatforms; ++i) 
            {
                char pbuf[100];

                if(clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(pbuf), pbuf, nullptr) != CL_SUCCESS)
                {
                    perror("clGetPlatformInfo failed.");
                    throw mlopenStatusInternalError;
                }

                platform = platforms[i];
                if (!strcmp(pbuf, "Advanced Micro Devices, Inc.")) 
                {
                    break;
                }
            }
        }

        /////////////////////////////////////////////////////////////////
        // Create an OpenCL context
        /////////////////////////////////////////////////////////////////
        cl_int status = 0;
        cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
        cl_context_properties* cprops = (nullptr == platform) ? nullptr : cps;
        ContextPtr result{clCreateContextFromType(cprops, CL_DEVICE_TYPE_GPU, nullptr, nullptr, &status)};
        if(status != CL_SUCCESS)
        {
            printf("status: %d",  status);
            perror("Error: Creating Handle. (clCreateContextFromType)");
            throw mlopenStatusInternalError;
        }
        return result;
    }
    void SetProfilingResult(cl_event& e)
    {
        size_t st, end;
        clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof(size_t), &st, NULL);
        clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof(size_t), &end, NULL);
        profiling_result = (end-st)*1e-6;
    }
};

Handle::Handle (int numStreams, mlopenAcceleratorQueue_t *streams) 
: impl(new HandleImpl())
{
    // TODO: Retain the queues
    for(int i=0;i<numStreams;i++) impl->queues.emplace_back(streams[i]);
}

Handle::Handle () 
: impl(new HandleImpl())
{
    /////////////////////////////////////////////////////////////////
    // Create an OpenCL context
    /////////////////////////////////////////////////////////////////

    impl->context = impl->create_context();
    /* First, get the size of device list data */
    size_t deviceListSize;
    if(clGetContextInfo(impl->context.get(), CL_CONTEXT_NUM_DEVICES, sizeof(size_t), &deviceListSize, nullptr) != CL_SUCCESS)
    {
        perror("Error: Getting Handle Info (device list size, clGetContextInfo)");
        throw mlopenStatusInternalError;
    }

    if(deviceListSize == 0)
    {
        perror("Error: No devices found.");
        throw mlopenStatusInternalError;
    }

    /////////////////////////////////////////////////////////////////
    // Detect OpenCL devices
    /////////////////////////////////////////////////////////////////
    std::vector<cl_device_id> devices(deviceListSize);

    /* Now, get the device list data */
    if(clGetContextInfo( impl->context.get(), CL_CONTEXT_DEVICES, deviceListSize*sizeof(cl_device_id), devices.data(), nullptr) != CL_SUCCESS)
    {
        perror("Error: Getting Handle Info (device list, clGetContextInfo)");
        throw mlopenStatusInternalError;
    }

    char deviceName[100];

    // Just using the first device as default
    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
    printf("Device Name: %s\n", deviceName);

    /////////////////////////////////////////////////////////////////
    // Create an OpenCL command queue
    /////////////////////////////////////////////////////////////////
    cl_int status = 0;
    impl->queues.emplace_back(clCreateCommandQueue(impl->context.get(), devices[0], CL_QUEUE_PROFILING_ENABLE, &status));
    if(status != CL_SUCCESS)
    {
        perror("Creating Command Queue. (clCreateCommandQueue)");
        throw mlopenStatusInternalError;
    } 
}

Handle::~Handle() 
{
}

mlopenAcceleratorQueue_t Handle::GetStream() const
{
    return impl->queues.front().get();
}

void Handle::EnableProfiling(bool enable)
{
    this->impl->enable_profiling = enable;
}

float Handle::GetKernelTime() const
{
    return this->impl->profiling_result;
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
    auto q = this->GetStream();
    OCLKernel obj = this->impl->cache.getKernel(q, 
            algorithm,
            network_config,
            program_name, 
            kernel_name,
            vld,
            vgd,
            params);
    if (this->impl->enable_profiling) return obj.Invoke(q, std::bind(&HandleImpl::SetProfilingResult, std::ref(*this->impl), std::placeholders::_1));
    else return obj.Invoke(q);
}

KernelInvoke Handle::GetKernel(
    const std::string& algorithm,
    const std::string& network_config)
{
    auto q = this->GetStream();
    OCLKernel obj = this->impl->cache.getKernel(
            algorithm,
            network_config);
    if (this->impl->enable_profiling) return obj.Invoke(q, std::bind(&HandleImpl::SetProfilingResult, std::ref(*this->impl), std::placeholders::_1));
    else return obj.Invoke(q);
}

void Handle::Finish() const
{
    clFinish(this->GetStream());
}

ManageDataPtr Handle::Create(int sz)
{
    cl_int status = CL_SUCCESS;
    auto result = ManageDataPtr{clCreateBuffer(impl->context.get(), CL_MEM_READ_ONLY, sz, nullptr, &status)};
    if (status != CL_SUCCESS) MLOPEN_THROW("OpenCL error creating buffer: " + std::to_string(status));
    return std::move(result);
}
ManageDataPtr& Handle::WriteTo(const void* data, ManageDataPtr& ddata, int sz)
{
    cl_int status = clEnqueueWriteBuffer(this->GetStream(), ddata.get(), CL_TRUE, 0, sz, data, 0, nullptr, nullptr);
    if (status != CL_SUCCESS) MLOPEN_THROW("OpenCL error writing to buffer: " + std::to_string(status));
    return ddata;
}

void Handle::ReadTo(void* data, const ManageDataPtr& ddata, int sz)
{
    auto status = clEnqueueReadBuffer(this->GetStream(), ddata.get(), CL_TRUE, 0, sz, data, 0, nullptr, nullptr);
    if (status != CL_SUCCESS) MLOPEN_THROW("OpenCL error reading from buffer: " + std::to_string(status));
}
}
