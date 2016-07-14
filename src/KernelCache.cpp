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

OCLKernel KernelCache::get(cl_command_queue &queue,
						 const std::string& algorithm,
						 const std::string& network_config,
                         const std::string& program_name,
                         const std::string& kernel_name,
						 const std::vector<size_t>& vld,
						 const std::vector<size_t>& vgd,
                         const std::string& params)
{
    return getInstance().getKernel(queue,
			algorithm,
			network_config,
			program_name,
			kernel_name, 
			vld, 
			vgd,
			params);
}

OCLKernel KernelCache::get( const std::string& algorithm,
						 const std::string& network_config)
{
    return getInstance().getKernel(algorithm, network_config);
}

OCLKernel KernelCache::getKernel(	const std::string& algorithm,
										const std::string& network_config) {

	std::pair<std::string, std::string> key = std::make_pair(algorithm, network_config);
#ifndef NDEBUG
	std::cout << "key: " << key.first <<" "<< key.second<< std::endl;
#endif

	auto kernel_iterator = kernel_map.find(key);
	if (kernel_iterator != kernel_map.end())
	{

		printf("kernel found\n");
		return kernel_iterator->second;
	}
	// TODO: Where should the default kernel be?
	else // return default kernel
	{
		printf("looking for default kernel (does not exist)\n");
		exit(0);
	}
}

OCLKernel KernelCache::getKernel(cl_command_queue &queue,
										const std::string& algorithm,
										const std::string& network_config,
                                        const std::string& program_name,
                                        const std::string& kernel_name,
										const std::vector<size_t>& vld,
										const std::vector<size_t>& vgd,
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

	std::pair<std::string, std::string> key = std::make_pair(algorithm, network_config);
#ifndef NDEBUG
    std::cout << "key: " << key.first << ',' << key.second << std::endl;
#endif

    auto kernel_iterator = kernel_map.find(key);
    if (kernel_iterator != kernel_map.end())
    {

		printf("kernel found\n");
        return kernel_iterator->second;
    }
    else //build program and compile the kernel;
    {

		printf("kernel not found\n");
		cl_program program = NULL;
        getProgram(program, queue, program_name, _params);
        if (program == nullptr) {
            std::cout << "Problem with getting program ["
                      << program_name << "] " << std::endl;
        //    return;
			//TODO: Return meaningful error
        }

        int status;

		cl_kernel kernel;
		status = CLHelper::CreateKernel(program, kernel, kernel_name);

        if (status != mlopenStatusSuccess) {
			std::cout << " Error creating OCL kernel\n";
        }
		
		OCLKernel _kernel(kernel, vld, vgd);
        kernel_map[key] = _kernel;
        return _kernel;
    }
}

mlopenStatus_t KernelCache::getProgram(cl_program &program,
		cl_command_queue &queue,
                                         const std::string& program_name,
                                         const std::string& params)
{
	mlopenStatus_t status;

 	status = CLHelper::LoadProgramFromSource(program, queue, program_name);
	if(status != mlopenStatusSuccess) {
		return status;
	}

	status = CLHelper::BuildProgram(program, queue, params);
	if(status != mlopenStatusSuccess) {
		return status;
	}

	return mlopenStatusSuccess;
}



KernelCache& KernelCache::getInstance()
{
    return singleton;
}

#endif // MLOpen_BACKEND_OPENCL
