/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017-2021 Advanced Micro Devices, Inc.
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

#include "miopen/common.hpp"
#include <miopen/config.h>
#include <miopen/handle.hpp>
#include <miopen/binary_cache.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle_lock.hpp>
#include <miopen/invoker.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/logger.hpp>
#include <miopen/timer.hpp>
#include <miopen/hipoc_program.hpp>

#if !MIOPEN_ENABLE_SQLITE_KERN_CACHE
#include <miopen/write_file.hpp>
#include <boost/filesystem/operations.hpp>
#endif

#include <miopen/filesystem.hpp>
#include <miopen/load_file.hpp>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <thread>
#include <miopen/nogpu/handle_impl.hpp>

#if MIOPEN_USE_HIPBLASLT
#include <hipblaslt/hipblaslt.h>
#endif

namespace miopen {

Handle::Handle(miopenAcceleratorQueue_t /* stream */) : Handle::Handle() {}

Handle::Handle() : impl(new HandleImpl())
{
    this->impl->target_properties.Init(this);
    MIOPEN_LOG_NQI(*this);
}

Handle::~Handle() {}

void Handle::SetStream(miopenAcceleratorQueue_t /* streamID */) const {}

void Handle::SetStreamFromPool(int) const {}
void Handle::ReserveExtraStreamsInPool(int) const {}

miopenAcceleratorQueue_t Handle::GetStream() const { return {}; }

void Handle::SetAllocator(miopenAllocatorFunction /* allocator */,
                          miopenDeallocatorFunction /* deallocator */,
                          void* /* allocatorContext */) const
{
}

void Handle::EnableProfiling(bool enable) const { this->impl->enable_profiling = enable; }

float Handle::GetKernelTime() const { return this->impl->profiling_result; }

Allocator::ManageDataPtr Handle::Create(std::size_t sz) const { return this->impl->allocator(sz); }

Allocator::ManageDataPtr&
Handle::WriteTo(const void* /* data */, Allocator::ManageDataPtr& ddata, std::size_t /* sz */) const
{
    return ddata;
}

void Handle::ReadTo(void* /* data */,
                    const Allocator::ManageDataPtr& /* ddata */,
                    std::size_t /* sz */) const
{
}

void Handle::ReadTo(void* /* data */, ConstData_t /* ddata */, std::size_t /* sz */) const {}

void Handle::Copy(ConstData_t /* src */, Data_t /* dest */, std::size_t /* size */) const {}

KernelInvoke Handle::AddKernel(const std::string& algorithm,
                               const std::string& network_config,
                               const fs::path& program_name,
                               const std::string& kernel_name,
                               const std::vector<size_t>& vld,
                               const std::vector<size_t>& vgd,
                               const std::string& params,
                               std::size_t cache_index,
                               const std::string& kernel_src) const
{
    auto obj = this->impl->cache.AddKernel(*this,
                                           algorithm,
                                           network_config,
                                           program_name,
                                           kernel_name,
                                           vld,
                                           vgd,
                                           params,
                                           cache_index,
                                           kernel_src);
    return this->Run(obj);
}

Invoker Handle::PrepareInvoker(const InvokerFactory& factory,
                               const std::vector<solver::KernelInfo>& kernels,
                               std::vector<Program>* programs_out) const
{
    std::vector<Kernel> built;
    built.reserve(kernels.size());
    if(programs_out != nullptr)
        programs_out->resize(kernels.size());

    for(auto i = 0; i < kernels.size(); ++i)
    {
        auto& k              = kernels[i];
        Program* program_out = programs_out != nullptr ? &(*programs_out)[i] : nullptr;

        MIOPEN_LOG_I2("Preparing kernel: " << k.kernel_name);

        const auto kernel = this->impl->cache.AddKernel(*this,
                                                        "",
                                                        "",
                                                        k.kernel_file,
                                                        k.kernel_name,
                                                        k.l_wk,
                                                        k.g_wk,
                                                        k.comp_options,
                                                        kernels.size(),
                                                        "",
                                                        program_out);
        built.push_back(kernel);
    }
    return factory(built);
}

void Handle::ClearKernels(const std::string& algorithm, const std::string& network_config) const
{
    this->impl->cache.ClearKernels(algorithm, network_config);
}
void Handle::ClearProgram(const fs::path& program_name, const std::string& params) const
{
    this->impl->cache.ClearProgram(program_name, params);
}

const std::vector<Kernel>& Handle::GetKernelsImpl(const std::string& algorithm,
                                                  const std::string& network_config) const
{
    return this->impl->cache.GetKernels(algorithm, network_config);
}

KernelInvoke Handle::Run(Kernel /*k*/, bool /*coop_launch*/) const { return {}; }

Program Handle::LoadProgram(const fs::path& program_name,
                            std::string params,
                            const std::string& kernel_src,
                            bool force_attach_binary) const
{
    std::ignore = force_attach_binary;

    if(program_name.extension() == ".mlir")
    {
        params += " -mcpu=" + this->GetTargetProperties().Name();
    }

    auto hsaco =
        miopen::LoadBinary(GetTargetProperties(), GetMaxComputeUnits(), program_name, params);
    auto pgmImpl     = std::make_shared<HIPOCProgramImpl>();
    pgmImpl->program = program_name;
    pgmImpl->target  = this->GetTargetProperties();
    auto p           = HIPOCProgram{};
    p.impl           = pgmImpl;
    if(hsaco.empty())
    {
        // avoid the constructor since it implicitly calls the HIP API
        pgmImpl->BuildCodeObject(params, kernel_src);
// auto p = HIPOCProgram{program_name, params, this->GetTargetProperties(), kernel_src};

// Save to cache
#if MIOPEN_ENABLE_SQLITE_KERN_CACHE
        miopen::SaveBinary(p.IsCodeObjectInMemory() ? p.GetCodeObjectBlob()
                                                    : miopen::LoadFile(p.GetCodeObjectPathname()),
                           this->GetTargetProperties(),
                           this->GetMaxComputeUnits(),
                           program_name,
                           params);
#else
        auto path = miopen::GetCachePath(false) / boost::filesystem::unique_path().string();
        if(p.IsCodeObjectInMemory())
            miopen::WriteFile(p.GetCodeObjectBlob(), path);
        else
            fs::copy_file(p.GetCodeObjectPathname(), path);
        miopen::SaveBinary(path, GetTargetProperties(), program_name, params);
#endif
    }
    else
    {
        pgmImpl->binary = std::vector<char>(hsaco.begin(), hsaco.end());
        // return HIPOCProgram{program_name, hsaco};
    }
    return p;
}

bool Handle::HasProgram(const fs::path& program_name, const std::string& params) const
{
    return this->impl->cache.HasProgram(program_name, params);
}

void Handle::AddProgram(Program prog, const fs::path& program_name, const std::string& params) const
{
    this->impl->cache.AddProgram(prog, program_name, params);
}

void Handle::Finish() const {}
void Handle::Flush() const {}

bool Handle::IsProfilingEnabled() const { return this->impl->enable_profiling; }

void Handle::ResetKernelTime() const { this->impl->profiling_result = 0.0; }
void Handle::AccumKernelTime(float curr_time) const { this->impl->profiling_result += curr_time; }

std::size_t Handle::GetLocalMemorySize() const { return this->impl->local_mem_size; }

std::size_t Handle::GetGlobalMemorySize() const { return this->impl->global_mem_size; }

std::size_t Handle::GetMaxComputeUnits() const { return this->impl->num_cu; }

std::size_t Handle::GetImage3dMaxWidth() const { return this->impl->img3d_max_width; }

std::size_t Handle::GetWavefrontWidth() const { return this->impl->warp_size; }

// No HIP API that could return maximum memory allocation size
// for a single object.
std::size_t Handle::GetMaxMemoryAllocSize()
{
    if(this->impl->max_mem_alloc_size == 0)
        return floor(0.85 * this->impl->global_mem_size);
    else
        return this->impl->max_mem_alloc_size;
}

bool Handle::CooperativeLaunchSupported() const { return false; }

const TargetProperties& Handle::GetTargetProperties() const
{
    return this->impl->target_properties;
}

std::string Handle::GetDeviceNameImpl() const { return this->impl->device_name; }
std::string Handle::GetDeviceName() const { return this->impl->target_properties.Name(); }

std::ostream& Handle::Print(std::ostream& os) const
{
    os << "stream: " << this->impl->stream << ", device_id: " << this->impl->device;
    return os;
}

shared<Data_t> Handle::CreateSubBuffer(Data_t data, std::size_t offset, std::size_t) const
{
    auto cdata = reinterpret_cast<char*>(data);
    return {cdata + offset, null_deleter{}};
}

shared<ConstData_t> Handle::CreateSubBuffer(ConstData_t data, std::size_t offset, std::size_t) const
{
    auto cdata = reinterpret_cast<const char*>(data);
    return {cdata + offset, null_deleter{}};
}

#if MIOPEN_USE_ROCBLAS

const rocblas_handle_ptr& Handle::rhandle() const { return this->impl->rhandle_; }

rocblas_handle_ptr Handle::CreateRocblasHandle(miopenAcceleratorQueue_t) const
{
    rocblas_handle x = nullptr;
    rocblas_create_handle(&x);
    auto result = rocblas_handle_ptr{x};
    return result;
}
#endif

#if MIOPEN_USE_HIPBLASLT
const hipblasLt_handle_ptr& Handle::HipblasLtHandle() const { return impl->hip_blasLt_handle; }

hipblasLt_handle_ptr Handle::CreateHipblasLtHandle() const
{
    hipblasLtHandle_t handle = nullptr;
    auto status              = hipblasLtCreate(&handle);
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        MIOPEN_THROW(miopenStatusInternalError, "hipBLASLt error encountered");
    }
    return hipblasLt_handle_ptr{handle};
}
#endif
} // namespace miopen
