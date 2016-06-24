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

#pragma once
#ifndef _KERNEL_CAHCE_HPP_
#define _KERNEL_CAHCE_HPP_

#if MLOpen_BACKEND_OPENCL

#include <string>
#include <unordered_map>
#include "MLOpen.h"
#include "OCLKernel.hpp"
/**
 * @brief The KernelCache class Build and cache kernels
 * singleton
 */
class KernelCache
{

public:

    typedef std::unordered_map<std::string, OCLKernel > KernelMap;

    static KernelCache& getInstance();

	static OCLKernel& get(cl_command_queue &queue,
                         const std::string& program_name,
                         const std::string& kernel_name,
                         const std::string& params = "");

    mlopenStatus_t getProgram(cl_program &program,
							cl_command_queue& queue,
                              const std::string& program_name,
                              const std::string& params = "");

	OCLKernel& getKernel(cl_command_queue &queue,
                         const std::string& program_name,
                         const std::string& kernel_name,
                         const std::string& params = "");


private:

    KernelMap kernel_map;

    KernelCache();

    static KernelCache singleton;
};

#endif // MLOpen_BACKEND_OPENCL

#endif //_KERNEL_CACHE_HPP_
