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


#include <mlopen/kernel_cache.hpp>
#include <mlopen/errors.hpp>

#include <iostream>
#include <iterator>

namespace mlopen {

#ifndef NDEBUG
static void dump_kernel_params(const std::string& program_name,
                        const std::string& kernel_name,
                        const std::vector<size_t>& vld,
                        const std::vector<size_t>& vgd,
                        const std::string& params)
{
    const char * keys[] = {
        "MLO_FILTER_SIZE0", "MLO_FILTER_SIZE1", "MLO_N_INPUTS", "MLO_N_OUTPUTS",
        "MLO_BATCH_SZ", "MLO_IN_HEIGHT", "MLO_IN_WIDTH", "MLO_OUT_HEIGHT", "MLO_OUT_WIDTH",
        "MLO_FLTR_SZ0", "MLO_FLTR_SZ1", "MLO_N_IN_CHNLS", "MLO_N_OUT_CHNLS"
    };
    int value[sizeof(keys) / sizeof(keys[0])] = { 0 };
    for (const char * p = params.c_str(); p && (p = strstr(p, "-D")) != nullptr; ) {
        p += (p[2] == ' ') ? 3 : 2;
        const char * q = strstr(p, "=");
        if (!q) break;
        q++;
        for (int i = 0; i < sizeof(keys) / sizeof(keys[0]); i++) {
            if (!strncmp(p, keys[i], strlen(keys[i]))) {
                value[i - ((i >= 9) ? 9 : 0)] = atoi(q);
                break;
            }
        }
    }
    //for(int i = 0; i < sizeof(keys)/sizeof(keys[0]); i++) printf("%s = %d\n", keys[i], value[i]);
    int msize = value[0] * value[1] * value[2] * value[3];
    int isize = value[4] * value[2] * value[5] * value[6];
    int osize = value[4] * value[3] * value[7] * value[8];
    std::cout << "runcl " << params << " src/Kernels/" << program_name << " -k " << kernel_name
              << " if#" << isize * 4 << ": if#" << msize * 4 << ": if#" << osize * 4 << ": iv#0 "
              << vgd[0] << "," << vgd[1] << "," << vgd[2] << "/"
              << vld[0] << "," << vld[1] << "," << vld[2] << std::endl;
}
#endif

Kernel KernelCache::GetKernel(const std::string& algorithm,
										const std::string& network_config) {

	std::pair<std::string, std::string> key = std::make_pair(algorithm, network_config);
#ifndef NDEBUG
	std::cout << "key: " << key.first <<" "<< key.second<< std::endl;
#endif

	auto kernel_iterator = kernel_map.find(key);
	if (kernel_iterator != kernel_map.end())
	{
		return kernel_iterator->second;
	}
	else
	{
        MLOPEN_THROW("looking for default kernel (does not exist): " + algorithm + ", " + network_config);
	}
}

Kernel KernelCache::GetKernel(Handle &h,
										const std::string& algorithm,
										const std::string& network_config,
                                        const std::string& program_name,
                                        const std::string& kernel_name,
										const std::vector<size_t>& vld,
										const std::vector<size_t>& vgd,
                                        std::string params)
{

    if (params.length() > 0)
    {
        // Ensure only one space after the -cl-std.
        // >1 space can cause an Apple compiler bug. See clSPARSE issue #141.
        if (params.at(0) != ' ') { params = " " + params; }
#ifndef NDEBUG
        dump_kernel_params(program_name, kernel_name, vld, vgd, params);
#endif
    }

	std::pair<std::string, std::string> key = std::make_pair(algorithm, network_config);
#ifndef NDEBUG
    std::cout << "key: " << key.first << ',' << key.second << std::endl;
#endif

    Program program;

    auto program_it = program_map.find(std::make_pair(program_name, params));
    if (program_it != program_map.end())
    {
        program = program_it->second;
    }
    else
    {
        program = h.LoadProgram(program_name, params);
		program_map[std::make_pair(program_name, params)] = program;
    }
    Kernel kernel{program, kernel_name, vld, vgd};
    if (!network_config.empty() && !algorithm.empty()) { kernel_map[key] = kernel; }
    return kernel;
}

KernelCache::KernelCache()
{}

} // namespace mlopen
