/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

#include <miopen/subbuffers.hpp>
#include <miopen/miopen.h>

#if MIOPEN_BACKEND_OPENCL
#include <miopen/handle.hpp>
#include <miopen/ocldeviceinfo.hpp>
#endif

#include <tuple>

namespace miopen {

std::size_t GetSubbufferAlignment(const miopen::Handle* handle)
{
#if MIOPEN_BACKEND_OPENCL
    constexpr const std::size_t inferred = (0x400 / 8) * 2;
    if(handle == nullptr)
        return inferred;
    return GetDeviceInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>(GetDevice(handle->GetStream()));
#elif MIOPEN_BACKEND_HIP
    std::ignore = handle;
    return 1;
#endif
}

} // namespace miopen
