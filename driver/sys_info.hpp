/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once
#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <string>

#ifdef __linux__
#include <unistd.h>
#include <sys/utsname.h>
#endif

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_runtime.h>
#endif

namespace RocmPerf {

#define HIP_CHECK(call)                                                         \
    {                                                                           \
        hipError_t err_ = call;                                                 \
        if(err_ != hipSuccess)                                                  \
        {                                                                       \
            std::cerr << "HIP error: " << hipGetErrorString(err_) << std::endl; \
        }                                                                       \
    }

class SysInfo
{
public:
    SysInfo(size_t major, size_t minor, size_t patch)
        : miopMajor(major), miopMinor(minor), miopPatch(patch)
    {
    }

    void ShowSysInfo()
    {
#ifdef __linux__
        // System information collection
        const std::string timestamp = GetTimestamp();
        const std::string hostname  = GetHostname();
        const std::string osInfo    = GetOsInfo();
        const std::string hipVer    = GetHipVersion();
        auto [cpuVendor, cpuModel]  = GetCpuInfo();
        const std::string ramSize   = GetRamSize();
        const std::string gpuInfo   = GetGpuInfo();
        const std::string amdgpuVer = GetAmdGpuVersion();

        // Format final output
        std::cout << "Timestamp: " << timestamp << "; "
                  << "Host Name: " << hostname << "; "
                  << "Operating System: " << osInfo << "; "
                  << "ROCm: " << hipVer << "; "
                  << "MIOpen Driver: " << miopMajor << "." << miopMinor << "." << miopPatch << "; "
                  << "CPU Vendor: " << cpuVendor << "; "
                  << "CPU Model: " << cpuModel << "; "
                  << "RAM Size: " << ramSize << "; "
                  << "GPU Model: " << gpuInfo << "; "
                  << "AMDGPU Driver: " << amdgpuVer << std::endl;
#else
        miopMajor;
        miopMinor;
        miopPatch;
#endif
    }

private:
    std::string GetTimestamp()
    {
        std::stringstream ss;
#ifdef __linux__
        auto now   = std::chrono::system_clock::now();
        auto now_c = std::chrono::system_clock::to_time_t(now);

        ss << std::put_time(std::gmtime(&now_c), "%Y-%m-%d %H:%M:%S UTC");
#endif
        return ss.str();
    }

    std::string GetHostname()
    {
        char name[256] = "";
#ifdef __linux__
        gethostname(name, sizeof(name));
#endif
        return name;
    }

    std::string GetOsInfo()
    {
#ifdef __linux__
        struct utsname buf;
        uname(&buf);
        return std::string(buf.sysname) + " " + buf.release;
#else
        return "unimplemented";
#endif
    }

    std::pair<std::string, std::string> GetCpuInfo()
    {
        std::string line;
        std::string vendor_id, model_name;
        std::set<std::string> physical_ids;
        std::string socket_info;
#ifdef __linux__
        std::ifstream cpuinfo("/proc/cpuinfo");
        while(getline(cpuinfo, line))
        {
            if(line.find("vendor_id") == 0 && vendor_id.empty())
            {
                vendor_id = line.substr(line.find(": ") + 2);
            }
            if(line.find("model name") == 0 && model_name.empty())
            {
                model_name = line.substr(line.find(": ") + 2);
            }
            if(line.find("physical id") == 0)
            {
                physical_ids.insert(line.substr(line.find(": ") + 2));
            }
        }

        if(vendor_id.find("AuthenticAMD") != std::string::npos)
        {
            vendor_id        = "AMD";
            size_t start_pos = model_name.find("AMD ");
            if(start_pos != std::string::npos)
            {
                model_name = model_name.substr(start_pos + 4);
            }
            std::istringstream iss(model_name);
            std::string part1, part2;
            iss >> part1 >> part2;
            model_name = part1 + " " + part2;
        }
        else if(vendor_id.find("GenuineIntel") != std::string::npos)
        {
            vendor_id = "Intel";
        }
        else
        {
            vendor_id = "unknown";
        }

        // Format socket count
        socket_info = physical_ids.empty() ? "" : std::to_string(physical_ids.size()) + " x ";
        return {vendor_id, socket_info + model_name};
#else
        return {"unimplemented", "unimplemented"};
#endif
    }

    std::string GetRamSize()
    {
#ifdef __linux__
        std::ifstream meminfo("/proc/meminfo");
        std::string line;
        while(getline(meminfo, line))
        {
            if(line.find("MemTotal") == 0)
            {
                size_t start = line.find(":") + 2;
                size_t end   = line.find(" kB");
                long kb      = std::stol(line.substr(start, end - start));
                return std::to_string(kb / (1024 * 1024)) + " GB";
            }
        }
#endif
        return "Unknown";
    }

    std::string GetAmdGpuVersion()
    {
        std::string version = "0.0.0";
#ifdef __linux__
        std::ifstream amdgpuVer("/sys/module/amdgpu/version");
        if(amdgpuVer.is_open())
        {
            std::getline(amdgpuVer, version);
        }
#endif

        return version;
    }

    std::string GetHipVersion()
    {
        int runtime_version = 0;
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
        HIP_CHECK(hipRuntimeGetVersion(&runtime_version));
#endif
        const int patch = runtime_version % 100000;
        runtime_version = runtime_version / 100000;

        const int major = runtime_version / 100;
        const int minor = runtime_version % 100;
        return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
    }

    std::string GetGpuInfo()
    {
        std::string result;
        int deviceCount = 0;
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
        HIP_CHECK(hipGetDeviceCount(&deviceCount));
#endif
        if(deviceCount < 1)
        {
            result = "None";
        }
        else
        {
            std::map<std::string, int> gpuList;
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
            for(int i = 0; i < deviceCount; i++)
            {
                hipDeviceProp_t props;
                HIP_CHECK(hipGetDeviceProperties(&props, i));
                gpuList[props.name]++;
            }
#endif
            for(const auto& [name, count] : gpuList)
            {
                if(!result.empty())
                    result += ", ";
                result += std::to_string(count) + " x " + name;
            }
        }

        return result;
    }

private:
    size_t miopMajor{};
    size_t miopMinor{};
    size_t miopPatch{};
};
} // namespace RocmPerf
