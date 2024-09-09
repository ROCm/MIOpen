/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include "gtest_common.hpp"

std::ostream& operator<<(std::ostream& os, const DevDescription& dd)
{
    return os << dd.name << "(" << dd.cu_cnt << ")";
}

MockHandle::MockHandle(const DevDescription& dev_description) : dev_descr{dev_description} {}

std::string MockHandle::GetDeviceName() const { return std::string{dev_descr.name}; }

std::size_t MockHandle::GetMaxComputeUnits() const { return dev_descr.cu_cnt; }

std::size_t MockHandle::GetMaxMemoryAllocSize() { return std::numeric_limits<std::size_t>::max(); }

bool MockHandle::CooperativeLaunchSupported() const { return false; }

Gpu GetDevGpuType()
{
    const auto dev_name = get_handle().GetDeviceName();

    static const auto dev = [&] {
        if(dev_name == "gfx900")
            return Gpu::gfx900;
        else if(dev_name == "gfx906")
            return Gpu::gfx906;
        else if(dev_name == "gfx908")
            return Gpu::gfx908;
        else if(dev_name == "gfx90a")
            return Gpu::gfx90A;
        else if(miopen::StartsWith(dev_name, "gfx94"))
            return Gpu::gfx94X;
        else if(miopen::StartsWith(dev_name, "gfx103"))
            return Gpu::gfx103X;
        else if(miopen::StartsWith(dev_name, "gfx110"))
            return Gpu::gfx110X;
        else if(miopen::StartsWith(dev_name, "gfx120"))
            return Gpu::gfx120X;
        else
            throw std::runtime_error("unknown_gpu");
    }();

    return dev;
}

const std::multimap<Gpu, DevDescription>& GetAllKnownDevices()
{
    static_assert(Gpu::gfx120X == Gpu::gfxLast);

    // https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html
    static const std::multimap<Gpu, DevDescription> known_devs = {
        // clang-format off
        {Gpu::gfx900,  {"gfx900",  64}},
        {Gpu::gfx906,  {"gfx906",  60}},
        {Gpu::gfx906,  {"gfx906",  64}},
        {Gpu::gfx908,  {"gfx908",  120}},
        {Gpu::gfx90A,  {"gfx90a",  104}},
        {Gpu::gfx90A,  {"gfx90a",  110}},
        {Gpu::gfx94X,  {"gfx940",  228}},
        {Gpu::gfx94X,  {"gfx941",  304}},
        {Gpu::gfx94X,  {"gfx942",  228}},
        {Gpu::gfx94X,  {"gfx942",  304}},
        {Gpu::gfx103X, {"gfx1030", 30}},
        {Gpu::gfx103X, {"gfx1030", 36}},
        {Gpu::gfx103X, {"gfx1030", 40}},
        {Gpu::gfx103X, {"gfx1031", 18}},
        {Gpu::gfx103X, {"gfx1031", 20}},
        {Gpu::gfx103X, {"gfx1032", 14}},
        {Gpu::gfx103X, {"gfx1032", 16}},
        {Gpu::gfx110X, {"gfx1100", 35}},
        {Gpu::gfx110X, {"gfx1100", 40}},
        {Gpu::gfx110X, {"gfx1100", 42}},
        {Gpu::gfx110X, {"gfx1100", 48}},
        {Gpu::gfx110X, {"gfx1101", 24}},
        {Gpu::gfx110X, {"gfx1101", 27}},
        {Gpu::gfx110X, {"gfx1101", 30}},
        {Gpu::gfx110X, {"gfx1102", 16}},
        {Gpu::gfx120X, {"gfx1201", 10000}}, //\todo 10000 is a dummy value, replace with real value.
        // clang-format on
    };
    return known_devs;
}

bool IsTestSupportedByDevice(Gpu supported_devs)
{
    if((supported_devs & GetDevGpuType()) != Gpu::None)
    {
        return true;
    }
    return false;
}
