
#include <miopen/handle.hpp>
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
        return "#include <hip/hip_runtime.h>\n  extern \"C\" { __global__ void write(int* data) { "
               "int num = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x; data[num] *= 2;}}\n";
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

void test_multithreads(kernel_type_t kern_type)
{
    auto&& h = get_handle();
    std::thread([&] { run2s(h, 16, kern_type); }).join();
    std::thread([&] { run2s(h, 32, kern_type); }).join();
    std::thread([&] { std::thread([&] { run2s(h, 64, kern_type); }).join(); }).join();
    run2s(h, 4, kern_type);
}

std::string WriteError(kernel_type_t kern_type)
{
    if(kern_type == miopenOpenCLKernelType)
        return "__kernel void write(__global int* data) { data[i] = 0; }\n";
    else if(kern_type == miopenHIPKernelType)
        return "#include <hip/hip_runtime.h>\n  extern \"C\" { __global__ void write(int* data) { "
               "data[num] *= 2;}}\n";
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
        return "#include <hip/hip_runtime.h>\n  extern \"C\" { __global__ void write(int* data) { "
               "}}\n";
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
    auto known_arch = {"gfx908", "gfx906", "gfx900", "gfx803"};
    auto this_arch  = h.GetDeviceName();
    EXPECT(std::any_of(
        known_arch.begin(), known_arch.end(), [&](std::string arch) { return arch == this_arch; }));
}

int main()
{
    auto&& h = get_handle();
    if(h.GetDeviceName() != "gfx803")
    {
        test_multithreads(miopenHIPKernelType);
        test_errors(miopenHIPKernelType);
// Warnings currently dont work in opencl
#if !MIOPEN_BACKEND_OPENCL
        test_warnings(miopenHIPKernelType);
#endif
    }
    test_multithreads(miopenOpenCLKernelType);
    test_errors(miopenOpenCLKernelType);
    test_arch_name();
// Warnings currently dont work in opencl
#if !MIOPEN_BACKEND_OPENCL
    test_warnings(miopenOpenCLKernelType);
#endif
}
