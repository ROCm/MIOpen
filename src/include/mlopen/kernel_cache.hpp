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
#ifndef GUARD_MLOPEN_KERNEL_CACHE_HPP_
#define GUARD_MLOPEN_KERNEL_CACHE_HPP_

#include <string>
#include <unordered_map>
#include <mlopen.h>
#include <mlopen/oclkernel.hpp>
#include <mlopen/clhelper.hpp>

namespace mlopen {

struct SimpleHash {
	size_t operator()(const std::pair<std::string, std::string>& p) const {
		using std::hash;
		return (hash<std::string>()(p.first) ^ hash<std::string>()(p.second));
	}
};

/**
 * @brief The KernelCache class Build and cache kernels
 * 
 */
class KernelCache
{

public:

	typedef std::pair<std::string, std::string> Key;
    typedef std::unordered_map< Key, OCLKernel, SimpleHash > KernelMap;


	OCLKernel GetKernel(cl_command_queue &queue,
						 const std::string& algorithm,
						 const std::string& network_config,
                         const std::string& program_name,
                         const std::string& kernel_name,
						 const std::vector<size_t>& ldims,
						 const std::vector<size_t>& gdims,
                        std::string params = "");
	
	OCLKernel GetKernel( const std::string& algorithm,
						 const std::string& network_config);



    KernelCache();
    
private:

    KernelMap kernel_map;
};

}

#endif // GUARD_MLOPEN_KERNEL_CACHE_HPP_
