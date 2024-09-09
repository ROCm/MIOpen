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
#ifndef GUARD_MIOPEN_HANDLE_HPP_
#define GUARD_MIOPEN_HANDLE_HPP_

#include <miopen/config.h>
#include <miopen/kernel_info.hpp>
#include <miopen/common.hpp>
#include <miopen/invoker_cache.hpp>
#include <miopen/kernel.hpp>
#include <miopen/miopen.h>
#include <miopen/names.hpp>
#include <miopen/object.hpp>
#include <miopen/allocator.hpp>
#include <miopen/simple_hash.hpp>
#include <miopen/solver_id.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/target_properties.hpp>

#include <boost/range/adaptor/transformed.hpp>

#include <cstdio>
#include <cstring>
#include <ios>
#include <sstream>
#include <memory>
#include <vector>
#include <unordered_map>

#if MIOPEN_USE_ROCBLAS
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-macros"
#define ROCBLAS_BETA_FEATURES_API 1
#pragma clang diagnostic pop
#include <miopen/manage_ptr.hpp>
#if MIOPEN_ROCBLAS_VERSION_FLAT < 2045000
#include <rocblas.h>
#else
#include <rocblas/rocblas.h>
#endif
#endif

#if MIOPEN_USE_HIPBLASLT
#include <hipblas/hipblas.h>

using hipblasLtHandle_t = void*;
extern "C" hipblasStatus_t hipblasLtDestroy(hipblasLtHandle_t handle);
#endif

namespace miopen {

struct HandleImpl;

#if MIOPEN_USE_ROCBLAS
using rocblas_handle_ptr = MIOPEN_MANAGE_PTR(rocblas_handle, rocblas_destroy_handle);
#endif

#if MIOPEN_USE_HIPBLASLT
using hipblasLt_handle_ptr = MIOPEN_MANAGE_PTR(hipblasLtHandle_t, hipblasLtDestroy);
#endif

struct MIOPEN_EXPORT Handle : miopenHandle
{
    friend struct TargetProperties;

    Handle();
    Handle(miopenAcceleratorQueue_t stream);
    Handle(Handle&&) noexcept;
    virtual ~Handle();

    miopenAcceleratorQueue_t GetStream() const;
    void SetStream(miopenAcceleratorQueue_t streamID) const;
    void SetStreamFromPool(int streamID) const;
    void ReserveExtraStreamsInPool(int cnt) const;

    void SetAllocator(miopenAllocatorFunction allocator,
                      miopenDeallocatorFunction deallocator,
                      void* allocatorContext) const;

    void EnableProfiling(bool enable = true) const;

    void ResetKernelTime() const;
    void AccumKernelTime(float curr_time) const;

    float GetKernelTime() const;
    bool IsProfilingEnabled() const;

    KernelInvoke AddKernel(const std::string& algorithm,
                           const std::string& network_config,
                           const fs::path& program_name,
                           const std::string& kernel_name,
                           const std::vector<size_t>& vld,
                           const std::vector<size_t>& vgd,
                           const std::string& params,
                           std::size_t cache_index       = 0,
                           const std::string& kernel_src = "") const;

    void ClearKernels(const std::string& algorithm, const std::string& network_config) const;

    auto GetKernels(const std::string& algorithm, const std::string& network_config) const
    {
        return this->GetKernelsImpl(algorithm, network_config) |
               boost::adaptors::transformed([this](Kernel k) { return this->Run(k); });
    }
    KernelInvoke GetKernel(const std::string& algorithm, const std::string& network_config) const
    {
        auto ks = this->GetKernelsImpl(algorithm, network_config);
        if(ks.empty())
        {
            MIOPEN_THROW("looking for default kernel (does not exist): " + algorithm + ", " +
                         network_config);
        }
        return this->Run(ks.front());
    }

    KernelInvoke Run(Kernel k, bool coop_launch = false) const;
    const std::vector<Kernel>& GetKernelsImpl(const std::string& algorithm,
                                              const std::string& network_config) const;

    Program LoadProgram(const fs::path& program_name,
                        std::string params,
                        const std::string& kernel_src,
                        bool force_attach_binary = false) const;

    bool HasProgram(const fs::path& program_name, const std::string& params) const;
    void ClearProgram(const fs::path& program_name, const std::string& params) const;
    void AddProgram(Program prog, const fs::path& program_name, const std::string& params) const;

    void Finish() const;
    void Flush() const;

    std::size_t GetLocalMemorySize() const;
    std::size_t GetGlobalMemorySize() const;
    std::size_t GetImage3dMaxWidth() const;
    std::size_t GetWavefrontWidth() const;
    virtual std::size_t GetMaxComputeUnits() const;
    std::size_t GetMaxHardwareComputeUnits() const
    {
        const std::size_t num_cu = this->GetMaxComputeUnits();
        const std::string name   = this->GetDeviceName();
        return StartsWith(name, "gfx1") ? num_cu * 2 /* CUs per WGP */ : num_cu;
    }

    std::size_t m_MaxMemoryAllocSizeCached = 0;
    virtual std::size_t GetMaxMemoryAllocSize();
    virtual bool CooperativeLaunchSupported() const;

    virtual std::string GetDeviceName() const;
    const TargetProperties& GetTargetProperties() const;

private:
    std::string GetDeviceNameImpl() const;

public:
    std::ostream& Print(std::ostream& os) const;
    void Copy(ConstData_t src, Data_t dest, std::size_t size) const;

    Allocator::ManageDataPtr Create(std::size_t sz) const;
    Allocator::ManageDataPtr&
    WriteTo(const void* data, Allocator::ManageDataPtr& ddata, std::size_t sz) const;
    void ReadTo(void* data, const Allocator::ManageDataPtr& ddata, std::size_t sz) const;
    void ReadTo(void* data, ConstData_t ddata, std::size_t sz) const;
    shared<Data_t> CreateSubBuffer(Data_t data, std::size_t offset, std::size_t size) const;
#if MIOPEN_BACKEND_HIP
    shared<ConstData_t>
    CreateSubBuffer(ConstData_t data, std::size_t offset, std::size_t size) const;
#endif

    template <class T>
    Allocator::ManageDataPtr Create(std::size_t sz)
    {
        return this->Create(sz * sizeof(T));
    }

    template <class Container>
    Allocator::ManageDataPtr Write(const Container& c)
    {
        assert(!c.empty());
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

    template <class V>
    void ReadToVec(const Allocator::ManageDataPtr& ddata, V& output_vec)
    {
        using T = typename V::value_type;
        assert(ddata);
        assert(output_vec.size() > 0);
        this->ReadTo(output_vec.data(), ddata, output_vec.size() * sizeof(T));
    }

    static std::string GetDbBasename(const TargetProperties& target, size_t num_cu)
    {
        auto ret = target.DbId() + [&]() {
            std::ostringstream ss;
            if(num_cu <= 64)
                ss << '_' << num_cu;
            else
                ss << std::hex << num_cu;
            return std::string(ss.str());
        }();
        return ret;
    }

    std::string GetDbBasename() const
    {
        return GetDbBasename(GetTargetProperties(), GetMaxComputeUnits());
    }

    std::unique_ptr<HandleImpl> impl;
    std::unordered_map<std::string, std::vector<miopenConvSolution_t>> find_map;

    Invoker PrepareInvoker(const InvokerFactory& factory,
                           const std::vector<solver::KernelInfo>& kernels,
                           std::vector<Program>* programs_out = nullptr) const;

    void RegisterInvoker(const Invoker& invoker,
                         const NetworkConfig& config,
                         const std::string& solver,
                         const std::optional<AlgorithmName>& algo = std::nullopt)
    {
        invokers.Register({config, solver}, invoker);
        if(algo.has_value())
            SetAsFound1_0(config, *algo, solver);
    }

    void
    SetAsFound1_0(const NetworkConfig& config, const AlgorithmName& algo, const std::string& solver)
    {
        invokers.SetAsFound1_0(config, algo, solver);
    }

    std::optional<Invoker> GetInvoker(const NetworkConfig& config,
                                      const std::optional<solver::Id>& solver,
                                      const std::optional<AlgorithmName>& algo = std::nullopt) const
    {
        assert(solver || algo);
        assert(!(solver && algo));
        if(solver)
        {
            MIOPEN_LOG_I2("Returning an invoker for problem " << config.ToString() << " and solver "
                                                              << solver->ToString());
            return invokers[std::make_pair(config.ToString(), solver->ToString())];
        }

        if(!algo)
            MIOPEN_THROW(miopenStatusInternalError);

        MIOPEN_LOG_I2("Returning an invoker for problem " << config.ToString() << " and algorithm "
                                                          << algo->ToString());
        return invokers.GetFound1_0(config, *algo);
    }

    std::optional<std::string> GetFound1_0SolverId(const NetworkConfig& config,
                                                   const AlgorithmName& algo) const
    {
        return invokers.GetFound1_0SolverId(config, algo);
    }

#if MIOPEN_USE_ROCBLAS
    const rocblas_handle_ptr& rhandle() const;
#endif
#if MIOPEN_USE_HIPBLASLT
    const hipblasLt_handle_ptr& HipblasLtHandle() const;
#endif

private:
#if MIOPEN_USE_ROCBLAS
    rocblas_handle_ptr CreateRocblasHandle(miopenAcceleratorQueue_t streamID) const;
#endif
#if MIOPEN_USE_HIPBLASLT
    hipblasLt_handle_ptr CreateHipblasLtHandle() const;
#endif

    InvokerCache invokers;
};

inline std::ostream& operator<<(std::ostream& os, const Handle& handle) { return handle.Print(os); }

struct AutoEnableProfiling
{
    AutoEnableProfiling(const Handle& x) : h(x)
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
    const Handle& h;
    bool prev_state;
};

} // namespace miopen
MIOPEN_DEFINE_OBJECT(miopenHandle, miopen::Handle);

#endif // GUARD_MIOPEN_HANDLE_HPP_
