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

#include <miopen/config.h>
#include <miopen/handle.hpp>
#include <miopen/binary_cache.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/errors.hpp>
#include <miopen/gemm_geometry.hpp>
#include <miopen/handle_lock.hpp>
#include <miopen/invoker.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/logger.hpp>
#include <miopen/timer.hpp>
#include <miopen/hipoc_program.hpp>

#if !MIOPEN_ENABLE_SQLITE_KERN_CACHE
#include <miopen/write_file.hpp>
#endif

#include <boost/filesystem.hpp>
#include <miopen/handle_lock.hpp>
#include <miopen/load_file.hpp>
#include <miopen/gemm_geometry.hpp>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <thread>
#include <miopen/nogpu/handle_impl.hpp>
namespace miopen {

Handle::Handle(miopenAcceleratorQueue_t /* stream */) : Handle::Handle() {}

Handle::Handle() : impl(new HandleImpl())
{
    this->impl->target_properties.Init(this);
    MIOPEN_LOG_NQI(*this);
}

Handle::~Handle() {}

void Handle::SetStream(miopenAcceleratorQueue_t /* streamID */) const {}

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

void Handle::Copy(ConstData_t /* src */, Data_t /* dest */, std::size_t /* size */) const {}

KernelInvoke Handle::AddKernel(const std::string& algorithm,
                               const std::string& network_config,
                               const std::string& program_name,
                               const std::string& kernel_name,
                               const std::vector<size_t>& vld,
                               const std::vector<size_t>& vgd,
                               const std::string& params,
                               std::size_t cache_index,
                               bool is_kernel_str,
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
                                           is_kernel_str,
                                           kernel_src);
    return this->Run(obj);
}

Invoker Handle::PrepareInvoker(const InvokerFactory& factory,
                               const std::vector<solver::KernelInfo>& kernels) const
{
    std::vector<Kernel> built;
    for(auto& k : kernels)
    {
        MIOPEN_LOG_I2("Preparing kernel: " << k.kernel_name);
        const auto kernel = this->impl->cache.AddKernel(*this,
                                                        "",
                                                        "",
                                                        k.kernel_file,
                                                        k.kernel_name,
                                                        k.l_wk,
                                                        k.g_wk,
                                                        k.comp_options,
                                                        kernels.size());
        built.push_back(kernel);
    }
    return factory(built);
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

KernelInvoke Handle::Run(Kernel /* k */) const { return {}; }

Program Handle::LoadProgram(const std::string& program_name,
                            std::string params,
                            bool is_kernel_str,
                            const std::string& kernel_src) const
{
    if((!miopen::EndsWith(program_name, ".mlir-cpp")) && (!miopen::EndsWith(program_name, ".mlir")))
    {
        params += " -mcpu=" + this->GetTargetProperties().Name();
    }

    auto hsaco       = miopen::LoadBinary(this->GetTargetProperties(),
                                    this->GetMaxComputeUnits(),
                                    program_name,
                                    params,
                                    is_kernel_str);
    auto pgmImpl     = std::make_shared<HIPOCProgramImpl>();
    pgmImpl->program = program_name;
    pgmImpl->target  = this->GetTargetProperties();
    auto p           = HIPOCProgram{};
    p.impl           = pgmImpl;
    if(hsaco.empty())
    {
        // avoid the constructor since it implicitly calls the HIP API
        pgmImpl->BuildCodeObject(params, is_kernel_str, kernel_src);
// auto p = HIPOCProgram{
//     program_name, params, is_kernel_str, this->GetTargetProperties(), kernel_src};

// Save to cache
#if MIOPEN_ENABLE_SQLITE_KERN_CACHE
        miopen::SaveBinary(p.IsCodeObjectInMemory()
                               ? p.GetCodeObjectBlob()
                               : miopen::LoadFile(p.GetCodeObjectPathname().string()),
                           this->GetTargetProperties(),
                           this->GetMaxComputeUnits(),
                           program_name,
                           params,
                           is_kernel_str);
#else
        auto path = miopen::GetCachePath(false) / boost::filesystem::unique_path();
        if(p.IsCodeObjectInMemory())
            miopen::WriteFile(p.GetCodeObjectBlob(), path);
        else
            boost::filesystem::copy_file(p.GetCodeObjectPathname(), path);
        miopen::SaveBinary(path, this->GetTargetProperties(), program_name, params, is_kernel_str);
#endif
    }
    else
    {
        pgmImpl->binary = std::vector<char>(hsaco.begin(), hsaco.end());
        // return HIPOCProgram{program_name, hsaco};
    }
    return p;
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

shared<Data_t> Handle::CreateSubBuffer(Data_t data, std::size_t offset, std::size_t)
{
    auto cdata = reinterpret_cast<char*>(data);
    return {cdata + offset, null_deleter{}};
}

shared<ConstData_t> Handle::CreateSubBuffer(ConstData_t data, std::size_t offset, std::size_t)
{
    auto cdata = reinterpret_cast<const char*>(data);
    return {cdata + offset, null_deleter{}};
}

#if MIOPEN_USE_ROCBLAS
rocblas_handle_ptr Handle::CreateRocblasHandle() const
{
    rocblas_handle x = nullptr;
    rocblas_create_handle(&x);
    auto result = rocblas_handle_ptr{x};
    return result;
}
#endif
} // namespace miopen
