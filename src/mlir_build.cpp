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

#include <miopen/config.h>
#include <miopen/hip_build_utils.hpp>
#include <miopen/logger.hpp>
#include <miopen/mlir_build.hpp>

#include <Miir.h>

#include <fstream>
#include <vector>

namespace miopen {
// Anonymous namespace
namespace {
/// Destroys handle in case of exception.
class AutoMiirHandle
{
    MiirHandle handle;

    public:
    AutoMiirHandle(const std::string& options) : handle(miirCreateHandle(options.c_str())) {}
    // Explicitly disable copy and assignment of the handle to avoid double-free risk
    AutoMiirHandle(const AutoMiirHandle&) = delete;
    void operator=(const AutoMiirHandle&) = delete;
    ~AutoMiirHandle() { miirDestroyHandle(handle); }
    MiirHandle operator()() { return handle; }
};

void check_miir_error(MiirStatus status, const std::string& miir_fn_name)
{
    switch(status)
    {
    case MIIR_SUCCESS: return;
    case MIIR_INVALID_PARAM: MIOPEN_THROW(miir_fn_name + " MIIR_INVALID_PARAM"); break;
    case MIIR_INVALID_MODULE: MIOPEN_THROW(miir_fn_name + " MIIR_INVALID_MODULE"); break;
    case MIIR_BUILD_FAILURE: MIOPEN_THROW(miir_fn_name + " MIIR_BUILD_FAILURE"); break;
    default: MIOPEN_THROW(miir_fn_name + " <UNKNOWN ERROR>");
    }
}
} // namespace

void MiirGenLaunchParams(const std::string& params, size_t& local_size, size_t& global_size)
{
    AutoMiirHandle handle(params);
    auto status = miirLowerTuningParams(handle());
    check_miir_error(status, "miirLowerTuningParams");
    miirGetExecutionDims(handle(), &global_size, &local_size);
    check_miir_error(status, "miirGetExecutionDims");
}

bool MiirIsConfigApplicable(const std::string& params)
{
    AutoMiirHandle handle(params);
    return MIIR_SUCCESS == miirLowerTuningParams(handle());
}

void MiirGenBin(const std::string& params, std::vector<char>& buffer)
{
    AutoMiirHandle handle(params);
    miirLowerBin(handle());

    size_t size = 0;
    auto status = miirBufferGet(handle(), nullptr, &size);
    check_miir_error(status, "miirBufferGet");
    buffer.resize(size);
    status = miirBufferGet(handle(), buffer.data(), &size);
    check_miir_error(status, "miirBufferGet");
}

int MiirGetKernelCount(const std::string& params)
{
    AutoMiirHandle handle(params);
    const auto kernel_count = miirGetKernelCount(handle());
    if(kernel_count < 1)
        MIOPEN_THROW("miirGetKernelCount invalid count: " + std::to_string(kernel_count));
    return kernel_count;
}

} // namespace miopen
