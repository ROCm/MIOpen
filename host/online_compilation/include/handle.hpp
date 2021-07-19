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
#ifndef GUARD_OLC_HANDLE_HPP_
#define GUARD_OLC_HANDLE_HPP_

#include <kernel.hpp>
#include <stringutils.hpp>
#include <target_properties.hpp>

#include <boost/range/adaptor/transformed.hpp>

#include <cstdio>
#include <cstring>
#include <ios>
#include <sstream>
#include <memory>
#include <vector>
#include <unordered_map>

namespace olCompile {

struct HandleImpl;

struct Handle
{
    friend struct TargetProperties;

    Handle();
    Handle(hipStream_t stream);
    Handle(Handle&&) noexcept;
    ~Handle();

    hipStream_t GetStream() const;
    void SetStream(hipStream_t streamID) const;

    KernelInvoke AddKernel(const std::string& algorithm,
                           const std::string& network_config,
                           const std::string& program_name,
                           const std::string& kernel_name,
                           const std::vector<size_t>& vld,
                           const std::vector<size_t>& vgd,
                           const std::string& params,
                           std::size_t cache_index = 0) const;

    bool HasKernel(const std::string& algorithm, const std::string& network_config) const;

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
            throw std::runtime_error("looking for default kernel (does not exist): " + algorithm +
                                     ", " + network_config);
        }
        return this->Run(ks.front());
    }

    KernelInvoke Run(Kernel k) const;

    Program LoadProgram(const std::string& program_name, std::string params) const;

    bool HasProgram(const std::string& program_name, const std::string& params) const;

    void AddProgram(Program prog, const std::string& program_name, const std::string& params) const;

    void Finish() const;

    std::size_t GetLocalMemorySize() const;
    std::size_t GetGlobalMemorySize() const;
    std::size_t GetWavefrontWidth() const;
    std::size_t GetMaxComputeUnits() const;
    std::size_t GetMaxHardwareComputeUnits() const
    {
        std::size_t num_cu = this->GetMaxComputeUnits();
        std::string name   = this->GetDeviceName();
        return StartsWith(name, "gfx1") ? num_cu * 2 /* CUs per WGP */ : num_cu;
    }

    std::string GetDeviceName() const;
    const TargetProperties& GetTargetProperties() const;

    private:
    std::string GetDeviceNameImpl() const;
    const std::vector<Kernel>& GetKernelsImpl(const std::string& algorithm,
                                              const std::string& network_config) const;

    public:
    std::ostream& Print(std::ostream& os) const;

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
};

inline std::ostream& operator<<(std::ostream& os, const Handle& handle) { return handle.Print(os); }

} // namespace olCompile

#endif // GUARD_OLC_HANDLE_HPP_
