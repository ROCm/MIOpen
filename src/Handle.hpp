#ifndef _HANDLE_HPP_
#define _HANDLE_HPP_

#include "MLOpen.h"
#include <vector>

// TODO: Should be here and not in MLOpen.h
#if 0
#if MLOpen_BACKEND_OPENCL
#include <CL/cl.h>
typedef cl_command_queue mlopenStream_t;

#elif MLOpen_BACKEND_HIP
#include <hip_runtime.h>
typedef hipStream_t mlopenStream_t;

#endif // OpenCL or HIP
#endif

template <typename Stream>
struct mlopenContext {
	
	mlopenContext();
	mlopenContext(mlopenStream_t stream);
	~mlopenContext();

	mlopenStatus_t CreateDefaultStream();
	mlopenStatus_t SetStream(mlopenStream_t stream);
	mlopenStatus_t GetStream(mlopenStream_t *stream) const;

	std::vector<mlopenStream_t> _streams;
};

template <typename Stream>
mlopenContext<Stream>::mlopenContext (Stream stream) {
	_streams.push_back(stream);
}

template <typename Stream>
mlopenStatus_t mlopenContext<Stream>::SetStream (Stream stream) {
	_streams.push_back(stream);
	return mlopenStatusSuccess;
}

template <typename Stream>
mlopenStatus_t mlopenContext<Stream>::GetStream (Stream *stream) {
	*stream = _streams.back();
	return mlopenStatusSuccess;
}

#if MLOpen_BACKEND_OPENCL
template<>
mlopenStatus_t mlopenContext<cl_command_queue> CreateDefaultStream() {
	cl_int status = 0;
    size_t deviceListSize;
	unsigned int i;

	// TODO: Change errors to CL errors
    cl_uint numPlatforms;
    cl_platform_id platform = NULL;
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if(status != CL_SUCCESS)
    {
        fprintf(stderr,"clGetPlatformIDs failed. %u",numPlatforms);
        return 1;
    }
    if (0 < numPlatforms) 
    {
		cl_platform_id* platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
        status = clGetPlatformIDs(numPlatforms, platforms, NULL);
        if(status != CL_SUCCESS)
        {
            perror( "clGetPlatformIDs failed.2");
            return 1;
        }
        for (i = 0; i < numPlatforms; ++i) 
        {
            char pbuf[100];
            status = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(pbuf), pbuf, NULL);

            if(status != CL_SUCCESS)
            {
                perror("clGetPlatformInfo failed.");
                return 1;
            }

            platform = platforms[i];
            if (!strcmp(pbuf, "Advanced Micro Devices, Inc.")) 
            {
                break;
            }
        }
		free(platforms);
    }

    /////////////////////////////////////////////////////////////////
    // Create an OpenCL context
    /////////////////////////////////////////////////////////////////

    cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
    cl_context_properties* cprops = (NULL == platform) ? NULL : cps;
    cl_context context = clCreateContextFromType(cprops, CL_DEVICE_TYPE_GPU, NULL, NULL, &status);
    if(status != CL_SUCCESS)
    {
        printf("status: %d",  status);
        perror("Error: Creating Context. (clCreateContextFromType)");
        return 1;
    }
    /* First, get the size of device list data */
    status = clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(size_t), &deviceListSize, NULL);
    if(status != CL_SUCCESS)
    {
        perror("Error: Getting Context Info (device list size, clGetContextInfo)");
        return 1;
    }

    if(deviceListSize == 0)
    {
        perror("Error: No devices found.");
        return 1;
    }

    /////////////////////////////////////////////////////////////////
    // Detect OpenCL devices
    /////////////////////////////////////////////////////////////////
    devices = (cl_device_id *)malloc(sizeof(cl_device_id));

    /* Now, get the device list data */
    status = clGetContextInfo( context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), devices, NULL);
    if(status != CL_SUCCESS)
    {
        perror("Error: Getting Context Info (device list, clGetContextInfo)");
        return 1;
    }

	char deviceName[100];

	clGetDeviceInfo(devices[0], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
	printf("Device Name: %s\n", deviceName);

    /////////////////////////////////////////////////////////////////
    // Create an OpenCL command queue
    /////////////////////////////////////////////////////////////////
    cl_command_queue = commandQueue = clCreateCommandQueueWithProperties(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &status);
    if(status != CL_SUCCESS)
    {
        perror("Creating Command Queue. (clCreateCommandQueue)");
        return 1;
    }
	
	SetStream(commandQueue);	
	return mlopenStatusSuccess;
}

#elif MLOpen_BACKEND_HIP

template<>
mlopenStatus_t mlopenContext<hipStream_t> CreateDefaultStream() {
	hipStream_t stream;
	hipStreamCreate(&stream);

	SetStream(stream);
	return mlopenStatusSuccess;
}
#endif // OpenCL vs HIP

#endif // _HANDLE_HPP_
