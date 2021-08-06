/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017-2020 Advanced Micro Devices, Inc.
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

#include <handle.hpp>

#include <binary_cache.hpp>
#include <env.hpp>
#include <kernel_cache.hpp>
#include <stringutils.hpp>
#include <target_properties.hpp>

#include <hipCheck.hpp>

#include <write_file.hpp>

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <thread>

OLC_DECLARE_ENV_VAR(OLC_DEVICE_CU)

namespace online_compile {

std::size_t GetAvailableMemory()
{
    size_t free, total;
    MY_HIP_CHECK(hipMemGetInfo(&free, &total));
    return free;
}

int get_device_id() // Get random device
{
    int device;

    MY_HIP_CHECK(hipGetDevice(&device));

    return device;
}

void set_device(int id) { MY_HIP_CHECK(hipSetDevice(id)); }

int set_default_device()
{
    int n;

    MY_HIP_CHECK(hipGetDeviceCount(&n));

    // Pick device based on process id
    auto pid = ::getpid();
    assert(pid > 0);
    set_device(pid % n);
    return (pid % n);
}

struct HandleImpl
{
    using StreamPtr = std::shared_ptr<typename std::remove_pointer<hipStream_t>::type>;

    HandleImpl() {}

    StreamPtr create_stream()
    {
        hipStream_t result;

        MY_HIP_CHECK(hipStreamCreate(&result));

        return StreamPtr{result, &hipStreamDestroy};
    }

    static StreamPtr reference_stream(hipStream_t s) { return StreamPtr{s, null_deleter{}}; }

    std::string get_device_name() const
    {
        hipDeviceProp_t props;

        MY_HIP_CHECK(hipGetDeviceProperties(&props, device));

        const std::string name(props.gcnArchName);
        return name;
    }

    StreamPtr stream = nullptr;
    int device       = -1;
    KernelCache cache;
    TargetProperties target_properties;
};

Handle::Handle(hipStream_t stream) : impl(new HandleImpl())
{
    this->impl->device = get_device_id();

    if(stream == nullptr)
        this->impl->stream = HandleImpl::reference_stream(nullptr);
    else
        this->impl->stream = HandleImpl::reference_stream(stream);

    this->impl->target_properties.Init(this);
}

Handle::Handle() : impl(new HandleImpl())
{
    this->impl->device = get_device_id();
    this->impl->stream = HandleImpl::reference_stream(nullptr);

    this->impl->target_properties.Init(this);
}

Handle::~Handle() {}

void Handle::SetStream(hipStream_t streamID) const
{
    this->impl->stream = HandleImpl::reference_stream(streamID);

    this->impl->target_properties.Init(this);
}

hipStream_t Handle::GetStream() const { return impl->stream.get(); }

KernelInvoke Handle::AddKernel(const std::string& algorithm,
                               const std::string& network_config,
                               const std::string& program_name,
                               const std::string& kernel_name,
                               const std::vector<size_t>& vld,
                               const std::vector<size_t>& vgd,
                               const std::string& params,
                               std::size_t cache_index) const
{

    auto obj = this->impl->cache.AddKernel(
        *this, algorithm, network_config, program_name, kernel_name, vld, vgd, params, cache_index);
    return this->Run(obj);
}

void Handle::ClearKernels(const std::string& algorithm, const std::string& network_config) const
{
    this->impl->cache.ClearKernels(algorithm, network_config);
}

const std::vector<Kernel>& Handle::GetKernelsImpl(const std::string& algorithm,
                                                  const std::string& network_config) const
{
    return this->impl->cache.GetKernels(algorithm, network_config);
}

bool Handle::HasKernel(const std::string& algorithm, const std::string& network_config) const
{
    return this->impl->cache.HasKernels(algorithm, network_config);
}

KernelInvoke Handle::Run(Kernel k) const { return k.Invoke(this->GetStream()); }

Program Handle::LoadProgram(const std::string& program_name, std::string params) const
{
    if((!online_compile::EndsWith(program_name, ".mlir-cpp")) &&
       (!online_compile::EndsWith(program_name, ".mlir")))
    {
        params += " -mcpu=" + this->GetTargetProperties().Name();
    }

    auto hsaco = online_compile::LoadBinary(
        this->GetTargetProperties(), this->GetMaxComputeUnits(), program_name, params);
    if(hsaco.empty())
    {
        auto p = HIPOCProgram{program_name, params, this->GetTargetProperties()};

        auto path = online_compile::GetCachePath() / boost::filesystem::unique_path();
        if(p.IsCodeObjectInMemory())
            online_compile::WriteFile(p.GetCodeObjectBlob(), path);
        else
            boost::filesystem::copy_file(p.GetCodeObjectPathname(), path);
        online_compile::SaveBinary(path, this->GetTargetProperties(), program_name, params);

        return p;
    }
    else
    {
        return HIPOCProgram{program_name, hsaco};
    }
}

bool Handle::HasProgram(const std::string& program_name, const std::string& params) const
{
    return this->impl->cache.HasProgram(program_name, params);
}

void Handle::AddProgram(Program prog,
                        const std::string& program_name,
                        const std::string& params) const
{
    this->impl->cache.AddProgram(prog, program_name, params);
}

void Handle::Finish() const { MY_HIP_CHECK(hipStreamSynchronize(this->GetStream())); }

std::size_t Handle::GetLocalMemorySize() const
{
    int result;

    MY_HIP_CHECK(hipDeviceGetAttribute(
        &result, hipDeviceAttributeMaxSharedMemoryPerBlock, this->impl->device));

    return result;
}

std::size_t Handle::GetGlobalMemorySize() const
{
    size_t result;

    MY_HIP_CHECK(hipDeviceTotalMem(&result, this->impl->device));

    return result;
}

std::size_t Handle::GetMaxComputeUnits() const
{
    int result;
    const char* const num_cu = online_compile::GetStringEnv(OLC_DEVICE_CU{});
    if(num_cu != nullptr && strlen(num_cu) > 0)
    {
        return boost::lexical_cast<std::size_t>(num_cu);
    }

    MY_HIP_CHECK(
        hipDeviceGetAttribute(&result, hipDeviceAttributeMultiprocessorCount, this->impl->device));

    return result;
}

std::size_t Handle::GetWavefrontWidth() const
{
    hipDeviceProp_t props{};

    MY_HIP_CHECK(hipGetDeviceProperties(&props, this->impl->device));

    auto result = static_cast<size_t>(props.warpSize);
    return result;
}

std::string Handle::GetDeviceNameImpl() const { return this->impl->get_device_name(); }

std::string Handle::GetDeviceName() const { return this->impl->target_properties.Name(); }

const TargetProperties& Handle::GetTargetProperties() const
{
    return this->impl->target_properties;
}

std::ostream& Handle::Print(std::ostream& os) const
{
    os << "stream: " << this->impl->stream << ", device_id: " << this->impl->device;
    return os;
}

} // namespace online_compile
