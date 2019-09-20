/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_CONTEXT_HPP_
#define GUARD_MIOPEN_CONTEXT_HPP_

#include <cstdio>
#include <cstring>
#include <ios>
#include <sstream>
#include <memory>
#include <miopen/config.h>
#include <miopen/common.hpp>
#include <miopen/kernel.hpp>
#include <miopen/miopen.h>
#include <miopen/object.hpp>
#include <miopen/allocator.hpp>
#include <miopen/simple_hash.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <vector>
#include <unordered_map>

#if MIOPEN_USE_ROCBLAS
#include <miopen/manage_ptr.hpp>
#include <rocblas.h>
#endif

namespace miopen {

struct HandleImpl;
#if MIOPEN_USE_MIOPENGEMM
struct GemmGeometry;
using GemmKey = std::pair<std::string, std::string>;
#endif

#if MIOPEN_USE_ROCBLAS
using rocblas_handle_ptr = MIOPEN_MANAGE_PTR(rocblas_handle, rocblas_destroy_handle);
#endif

struct Handle : miopenHandle
{

    Handle();
    Handle(miopenAcceleratorQueue_t stream);
    Handle(Handle&&) noexcept;
    ~Handle();

    miopenAcceleratorQueue_t GetStream() const;
    void SetStream(miopenAcceleratorQueue_t streamID) const;

    void SetAllocator(miopenAllocatorFunction allocator,
                      miopenDeallocatorFunction deallocator,
                      void* allocatorContext) const;

    void EnableProfiling(bool enable = true);

    void ResetKernelTime();
    void AccumKernelTime(float curr_time);

    float GetKernelTime() const;
    bool IsProfilingEnabled() const;

    KernelInvoke AddKernel(const std::string& algorithm,
                           const std::string& network_config,
                           const std::string& program_name,
                           const std::string& kernel_name,
                           const std::vector<size_t>& vld,
                           const std::vector<size_t>& vgd,
                           const std::string& params,
                           std::size_t cache_index       = 0,
                           bool is_kernel_str            = false,
                           const std::string& kernel_src = "");

    bool HasKernel(const std::string& algorithm, const std::string& network_config) const;

    void ClearKernels(const std::string& algorithm, const std::string& network_config);

    auto GetKernels(const std::string& algorithm, const std::string& network_config)
    {
        return this->GetKernelsImpl(algorithm, network_config) |
               boost::adaptors::transformed([this](Kernel k) { return this->Run(k); });
    }
    KernelInvoke GetKernel(const std::string& algorithm, const std::string& network_config)
    {
        auto ks = this->GetKernelsImpl(algorithm, network_config);
        if(ks.empty())
        {
            MIOPEN_THROW("looking for default kernel (does not exist): " + algorithm + ", " +
                         network_config);
        }
        return this->Run(ks.front());
    }

    KernelInvoke Run(Kernel k);
    const std::vector<Kernel>& GetKernelsImpl(const std::string& algorithm,
                                              const std::string& network_config);

    Program LoadProgram(const std::string& program_name,
                        std::string params,
                        bool is_kernel_str,
                        const std::string& kernel_src);

    void Finish() const;
    void Flush() const;

    std::size_t GetLocalMemorySize();
    std::size_t GetGlobalMemorySize();
    std::size_t GetMaxComputeUnits();
    std::size_t GetImage3dMaxWidth();
    std::size_t GetWavefrontWidth();

    std::size_t m_MaxMemoryAllocSizeCached = 0;
    std::size_t GetMaxMemoryAllocSize();

    std::string GetDeviceName();
    std::ostream& Print(std::ostream& os) const;

    void Copy(ConstData_t src, Data_t dest, std::size_t size);

    Allocator::ManageDataPtr Create(std::size_t sz);
    Allocator::ManageDataPtr&
    WriteTo(const void* data, Allocator::ManageDataPtr& ddata, std::size_t sz);
    void ReadTo(void* data, const Allocator::ManageDataPtr& ddata, std::size_t sz);
    shared<Data_t> CreateSubBuffer(Data_t data, std::size_t offset, std::size_t size);
#if MIOPEN_BACKEND_HIP
    shared<ConstData_t> CreateSubBuffer(ConstData_t data, std::size_t offset, std::size_t size);
#endif

    template <class T>
    Allocator::ManageDataPtr Create(std::size_t sz)
    {
        return this->Create(sz * sizeof(T));
    }

    template <class Container>
    Allocator::ManageDataPtr Write(const Container& c)
    {
        using type = typename Container::value_type;
        auto buf   = this->Create<type>(c.size());
        return std::move(
            this->WriteTo(reinterpret_cast<const void*>(c.data()), buf, c.size() * sizeof(type)));
    }

    template <class T>
    std::vector<T> Read(const Allocator::ManageDataPtr& ddata, std::size_t sz)
    {
        std::vector<T> result(sz);
        this->ReadTo(result.data(), ddata, sz * sizeof(T));
        return result;
    }

    std::string GetDbBasename()
    {
        const auto ret = GetDeviceName() + [&]() {
            std::ostringstream ss;
            const auto ncu = GetMaxComputeUnits();
            if(ncu <= 64)
                ss << '_' << ncu;
            else
                ss << std::hex << ncu;
            return std::string(ss.str());
        }();
        return ret;
    }

    std::unique_ptr<HandleImpl> impl;
    std::unordered_map<std::string, std::vector<miopenConvSolution_t>> find_map;
#if MIOPEN_USE_MIOPENGEMM
    std::unordered_map<GemmKey, std::unique_ptr<GemmGeometry>, SimpleHash> geo_map;
#endif

#if MIOPEN_USE_ROCBLAS
    rocblas_handle_ptr& rhandle() { return rhandle_; }

    private:
    rocblas_handle_ptr CreateRocblasHandle() const;

    rocblas_handle_ptr rhandle_;
#endif
};

inline std::ostream& operator<<(std::ostream& os, const Handle& handle) { return handle.Print(os); }

struct AutoEnableProfiling
{
    AutoEnableProfiling(Handle& x) : h(x)
    {
        prev_state = h.IsProfilingEnabled();
        h.EnableProfiling();
    }

    ~AutoEnableProfiling()
    {
        h.EnableProfiling(prev_state);
        h.ResetKernelTime();
    }

    private:
    Handle& h;
    bool prev_state;
};

} // namespace miopen
MIOPEN_DEFINE_OBJECT(miopenHandle, miopen::Handle);

#endif // GUARD_MIOPEN_CONTEXT_HPP_
