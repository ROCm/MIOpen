/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *all
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
#ifndef GUARD_FIN_HPP
#define GUARD_FIN_HPP

// using float16 = half_float::half;
#include "config.h"
#include "tensor.hpp"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <miopen/handle.hpp>
#include <numeric>
#include <vector>

using json = nlohmann::json;

#if FIN_BACKEND_OPENCL
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#elif FIN_BACKEND_HIP
#include <hip/hip_runtime_api.h>
#endif

namespace fin {
class Fin
{
    public:
    Fin() {}
    void Usage();
    std::string ParseBaseArg(const int argc, const char* argv[]);
    miopen::Handle& GetHandle()
    {
        static auto handle = miopen::Handle{};
        return handle;
    }
    miopenDataType_t GetDataType() { return data_type; }

#if FIN_BACKEND_OPENCL
    cl_command_queue& GetStream() { return q; }
#elif FIN_BACKEND_HIP
    hipStream_t& GetStream() { return q; }
#endif
    virtual ~Fin() {} // TODO: do we need this

    virtual int ProcessStep(const std::string& step_name) = 0;

    json output;

    protected:
    template <typename Tgpu>
    void InitDataType();
    miopenDataType_t data_type = miopenFloat; // the datatype passed in through the command line

#if FIN_BACKEND_OPENCL
    cl_command_queue q;
#elif FIN_BACKEND_HIP
    hipStream_t q;
#endif
};

// "std::is_same<Tgpu, float>{}" used to avoid "static_assert" compilation
// error,
// which occurs when the condition does not depend in any way on the template
// parameters.
template <typename Tgpu>
void Fin::InitDataType()
{
    static_assert(std::is_same<Tgpu, float>{}, "unsupported Tgpu");
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vs)
{
    os << "{ size: " << vs.size() << ", entries: ";
    for(auto& v : vs)
        os << v << " ";
    os << "}";
    return os;
}
} // namespace fin
#endif // GUARD_FIN_HPP
