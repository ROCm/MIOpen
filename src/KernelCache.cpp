/* ************************************************************************
 * Copyright 2015 Vratis, Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************ */

#if MLOpen_BACKEND_OPENCL

#include "KernelCache.hpp"

#include <iostream>
#include <iterator>

KernelCache KernelCache::singleton;

KernelCache::KernelCache()
{
    //we can add sth here which can be shared among all kernels;
}

OCLKernel& KernelCache::get(cl_command_queue &queue,
                         const std::string& program_name,
                         const std::string& kernel_name,
                         const std::string& params)
{
    return getInstance().getKernel(queue, program_name, kernel_name, params);
}


OCLKernel& KernelCache::getKernel(cl_command_queue &queue,
                                        const std::string& program_name,
                                        const std::string& kernel_name,
                                        const std::string& params)
{

	std::string _params = "";
    if (params.length() > 0)
    {
        // Ensure only one space after the -cl-std.
        // >1 space can cause an Apple compiler bug. See clSPARSE issue #141.
        if (params.at(0) != ' ')
            _params.append(" ");
        _params.append(params);
    }
    std::string key;
    key.append( "[" + program_name + "/"  + kernel_name + "]");
    key.append(_params);

#ifndef NDEBUG
    std::cout << "key: " << key << std::endl;
#endif

    auto kernel_iterator = kernel_map.find(key);
    if (kernel_iterator != kernel_map.end())
    {

		printf("kernel found\n");
#ifndef NDEBUG
//        std::cout << "kernel found: " << hash <<std::endl;
#endif
        return kernel_iterator->second;
    }
    else //build program and compile the kernel;
    {

		printf("kernel not found\n");
		cl_program program = NULL;
        getProgram(program, queue, program_name, _params);
        if (program == nullptr)
        {
            std::cout << "Problem with getting program ["
                      << program_name << "] " << std::endl;
        //    return;
			//TODO: Return meaningful error
        }

        cl_int status;

		cl_kernel kernel = clCreateKernel(program, 
				kernel_name.c_str(), 
				&status);

        if (status != CL_SUCCESS)
        {
            std::cout << "Problem with creating kernel ["
                      << kernel_name << "]" << std::endl;
          //  return;
			// TODO: Return meaningful error
        }

		OCLKernel _kernel(kernel);
        kernel_map[key] = _kernel;
        return _kernel;
    }
}

mlopenStatus_t KernelCache::getProgram(cl_program &program,
		cl_command_queue &queue,
                                         const std::string& program_name,
                                         const std::string& params)
{

    cl_int status;
	cl_context context;
	cl_device_id device;

	status = clGetCommandQueueInfo(queue,
			CL_QUEUE_CONTEXT, 
			sizeof(cl_context),
			&context, 
			NULL);
	// TODO: Check status

	// Stringify the kernel file
	char *source;
	size_t sourceSize;
	FILE *fp = fopen(program_name.c_str(), "rb");
	fseek(fp, 0, SEEK_END);
	sourceSize = ftell(fp);
	fseek(fp , 0, SEEK_SET);
	source = (char *)malloc(sourceSize * sizeof(char));
	fread(source, 1, sourceSize, fp);
	fclose(fp);

	program  = clCreateProgramWithSource(context, 
			1,
			(const char**)&source, 
			&sourceSize, 
			&status);
	// TODO: Check status

	status = clGetCommandQueueInfo(queue,
			CL_QUEUE_DEVICE, 
			sizeof(cl_device_id),
			&device, 
			NULL);
	// TODO: Check status


	/* create a cl program executable for all the devices specified */
    status = clBuildProgram(program, 
			1, &device, params.c_str(), 
			NULL, 
			NULL);
	// TODO: Check status
	
    if(status != CL_SUCCESS)
    {
        printf("Error: Building Program (clBuildProgram): %d", status);
        char * errorbuf = (char*)calloc(sizeof(char),1024*1024);
        size_t size;
        clGetProgramBuildInfo(program,
				device,
				CL_PROGRAM_BUILD_LOG, 
				1024*1024, 
				errorbuf,
				&size);

        printf("%s ", errorbuf);
		free(errorbuf);
    }

    return mlopenStatusSuccess;
}



KernelCache& KernelCache::getInstance()
{
    return singleton;
}

#endif // MLOpen_BACKEND_OPENCL
