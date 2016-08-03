#include "Handle.hpp"

#include <manage_ptr.hpp>


struct mlopenContextImpl
{

    typedef MLOPEN_MANAGE_PTR(mlopenAcceleratorQueue_t, clReleaseCommandQueue) AqPtr;
    typedef MLOPEN_MANAGE_PTR(cl_context, clReleaseContext) ContextPtr;

    ContextPtr context;
    std::vector<AqPtr> queues;

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
        ContextPtr context{clCreateContextFromType(cprops, CL_DEVICE_TYPE_GPU, nullptr, nullptr, &status)};
        if(status != CL_SUCCESS)
        {
            printf("status: %d",  status);
            perror("Error: Creating Context. (clCreateContextFromType)");
            throw mlopenStatusInternalError;
        }
        return context;
    }
};

mlopenContext::mlopenContext (int numStreams, mlopenAcceleratorQueue_t *streams) 
: impl(new mlopenContextImpl())
{
    // TODO: Retain the queues
    for(int i=0;i<numStreams;i++) impl->queues.emplace_back(streams[i]);
}

mlopenContext::mlopenContext () 
: impl(new mlopenContextImpl())
{
    /////////////////////////////////////////////////////////////////
    // Create an OpenCL context
    /////////////////////////////////////////////////////////////////

    impl->context = impl->create_context();
    /* First, get the size of device list data */
    size_t deviceListSize;
    if(clGetContextInfo(impl->context.get(), CL_CONTEXT_NUM_DEVICES, sizeof(size_t), &deviceListSize, nullptr) != CL_SUCCESS)
    {
        perror("Error: Getting Context Info (device list size, clGetContextInfo)");
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
        perror("Error: Getting Context Info (device list, clGetContextInfo)");
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

mlopenContext::~mlopenContext() {}

mlopenAcceleratorQueue_t mlopenContext::GetStream()
{
    return impl->queues.front().get();
}
