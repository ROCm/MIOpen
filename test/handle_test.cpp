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

/// SWDEV-257056?focusedCommentId=6654244&page=com.atlassian.jira.plugin.system.issuetabpanels%3Acomment-tabpanel#comment-6654244
/// \todo Create dedicated ticket and rename macro.
#define WORKAROUND_SWDEV_257056_PCH_MISSING_MACROS 1

#include <miopen/config.h>
#include <miopen/handle.hpp>
#include <miopen/execution_context.hpp>

#if WORKAROUND_SWDEV_257056_PCH_MISSING_MACROS
#include <miopen/hip_build_utils.hpp>
#endif

#include "get_handle.hpp"
#include <vector>
#include <thread>
#include "test.hpp"

enum kernel_type_t
{
    miopenHIPKernelType,
    miopenOpenCLKernelType
};

std::string Write2s(kernel_type_t kern_type)
{
    if(kern_type == miopenHIPKernelType)
        return "#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS\n"
               "#include <hip/hip_runtime.h>\n"
#if WORKAROUND_SWDEV_257056_PCH_MISSING_MACROS
               "#else\n"
               "#ifdef hipThreadIdx_x\n"
               "#undef hipThreadIdx_x\n"
               "#endif\n"
               "#define hipThreadIdx_x threadIdx.x\n"
               "\n"
               "#ifdef hipBlockDim_x\n"
               "#undef hipBlockDim_x\n"
               "#endif\n"
               "#define hipBlockDim_x blockDim.x\n"
               "\n"
               "#ifdef hipBlockIdx_x\n"
               "#undef hipBlockIdx_x\n"
               "#endif\n"
               "#define hipBlockIdx_x blockIdx.x\n"
#endif
               "#endif\n"
               "extern \"C\" {\n"
               "__global__ void write(int* data) {\n"
               "    int num = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;\n"
               "    data[num] *= 2;\n"
               "}\n"
               "}\n";
    else if(kern_type == miopenOpenCLKernelType)
        return "__kernel void write(__global int* data) { data[get_global_id(0)] *= 2; }\n";
    else
        MIOPEN_THROW("Unsupported kernel type");
}

void run2s(miopen::Handle& h, std::size_t n, kernel_type_t kern_type)
{
    std::vector<int> data_in(n, 1);
    auto data_dev = h.Write(data_in);
    if(kern_type == miopenOpenCLKernelType)
        h.AddKernel("GEMM", "", Write2s(miopenOpenCLKernelType), "write", {n, 1, 1}, {n, 1, 1}, "")(
            data_dev.get());
    else if(kern_type == miopenHIPKernelType)
        h.AddKernel("NoAlgo",
                    "",
                    "test_hip.cpp",
                    "write",
                    {n, 1, 1},
                    {n, 1, 1},
                    "",
                    0,
                    false,
                    Write2s(miopenHIPKernelType))(data_dev.get());
    else
        MIOPEN_THROW("Unsupported kernel type");
    std::fill(data_in.begin(), data_in.end(), 2);

    auto data_out = h.Read<int>(data_dev, n);
    CHECK(data_out == data_in);
}

void test_multithreads(kernel_type_t kern_type, const bool with_stream = false)
{
    auto&& h1 = get_handle();
    auto&& h2 = get_handle_with_stream(h1);
    std::thread([&] { run2s(with_stream ? h2 : h1, 16, kern_type); }).join();
    std::thread([&] { run2s(with_stream ? h2 : h1, 32, kern_type); }).join();
    std::thread([&] {
        std::thread([&] { run2s(with_stream ? h2 : h1, 64, kern_type); }).join();
    }).join();
    run2s(with_stream ? h2 : h1, 4, kern_type);
}

std::string WriteError(kernel_type_t kern_type)
{
    if(kern_type == miopenOpenCLKernelType)
        return "__kernel void write(__global int* data) { data[i] = 0; }\n";
    else if(kern_type == miopenHIPKernelType)
        return "#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS\n"
               "#include <hip/hip_runtime.h>\n"
               "#endif\n"
               "extern \"C\" {\n"
               "__global__ void write(int* data) {\n"
               "    data[num] *= 2;\n"
               "}\n"
               "}\n";
    else
        MIOPEN_THROW("Unsupported kernel type");
}

void test_errors(kernel_type_t kern_type)
{
    auto&& h = get_handle();
    if(kern_type == miopenOpenCLKernelType)
    {
        EXPECT(throws([&] {
            h.AddKernel("GEMM", "", WriteError(kern_type), "write", {1, 1, 1}, {1, 1, 1}, "");
        }));
        try
        {
            h.AddKernel("GEMM", "", WriteError(kern_type), "write", {1, 1, 1}, {1, 1, 1}, "");
        }
        catch(miopen::Exception& e)
        {
            EXPECT(!std::string(e.what()).empty());
        }
    }
    else if(kern_type == miopenHIPKernelType)
    {
        EXPECT(throws([&] {
            h.AddKernel("NoAlgo",
                        "",
                        "error_hip.cpp",
                        "write",
                        {1, 1, 1},
                        {1, 1, 1},
                        "",
                        0,
                        false,
                        WriteError(miopenHIPKernelType));
        }));
        try
        {
            h.AddKernel("NoAlgo",
                        "",
                        "error_hip.cpp",
                        "write",
                        {1, 1, 1},
                        {1, 1, 1},
                        "",
                        0,
                        false,
                        WriteError(miopenHIPKernelType));
        }
        catch(miopen::Exception& e)
        {
            EXPECT(!std::string(e.what()).empty());
        }
    }
}

std::string WriteNop(kernel_type_t kern_type)
{
    if(kern_type == miopenOpenCLKernelType)
        return "__kernel void write(__global int* data) {}\n";
    else if(kern_type == miopenHIPKernelType)
        return "#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS\n"
               "#include <hip/hip_runtime.h>\n"
               "#endif\n"
               "extern \"C\" {\n"
               "__global__ void write(int* data) {\n"
               "}\n"
               "}\n";
    else
        MIOPEN_THROW("Unsupported kernel type");
}

void test_warnings(kernel_type_t kern_type)
{
    auto&& h = get_handle();
#if MIOPEN_BUILD_DEV
    if(kern_type == miopenOpenCLKernelType)
        EXPECT(throws([&] {
            h.AddKernel("GEMM", "", WriteNop(kern_type), "write", {1, 1, 1}, {1, 1, 1}, "");
        }));
    else if(kern_type == miopenHIPKernelType)
        EXPECT(throws([&] {
            h.AddKernel("NoAlgo",
                        "",
                        "nop_hip.cpp",
                        "write",
                        {1, 1, 1},
                        {1, 1, 1},
                        "",
                        0,
                        false,
                        WriteNop(miopenHIPKernelType));
        }));
#else
    (void)kern_type;
    (void)h; // To silence warnings.
#endif
}

void test_arch_name()
{
    auto&& h        = get_handle();
    auto known_arch = {"gfx908", "gfx90a", "gfx906", "gfx900", "gfx803", "gfx1030", "gfx1031"};
    auto this_arch  = h.GetDeviceName();
    EXPECT(std::any_of(
        known_arch.begin(), known_arch.end(), [&](std::string arch) { return arch == this_arch; }));
}

int main()
{
    auto&& h = get_handle();
    if(h.GetDeviceName() != "gfx803" && miopen::IsHipKernelsEnabled())
    {
        test_multithreads(miopenHIPKernelType);
        test_errors(miopenHIPKernelType);
// Warnings currently dont work in opencl
#if !MIOPEN_BACKEND_OPENCL
        test_warnings(miopenHIPKernelType);
#endif
    }
    test_multithreads(miopenOpenCLKernelType);
    test_multithreads(miopenOpenCLKernelType, true);
    test_errors(miopenOpenCLKernelType);
    test_arch_name();
// Warnings currently dont work in opencl
#if !MIOPEN_BACKEND_OPENCL
    test_warnings(miopenOpenCLKernelType);
#endif
}
