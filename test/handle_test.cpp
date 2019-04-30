
#include <miopen/handle.hpp>
#include "get_handle.hpp"
#include <vector>
#include <thread>
#include "test.hpp"

std::string Write2s()
{
    return "__kernel void write(__global int* data) { data[get_global_id(0)] *= 2; }\n";
}

void run2s(miopen::Handle& h, std::size_t n)
{
    std::vector<int> data_in(n, 1);
    auto data_dev = h.Write(data_in);

    h.AddKernel("GEMM", "", Write2s(), "write", {n, 1, 1}, {n, 1, 1}, "")(data_dev.get());
    std::fill(data_in.begin(), data_in.end(), 2);

    auto data_out = h.Read<int>(data_dev, n);
    CHECK(data_out == data_in);
}

void test_multithreads()
{
    auto&& h = get_handle();
    std::thread([&] { run2s(h, 16); }).join();
    std::thread([&] { run2s(h, 32); }).join();
    std::thread([&] { std::thread([&] { run2s(h, 64); }).join(); }).join();
    run2s(h, 4);
}

std::string WriteError() { return "__kernel void write(__global int* data) { data[i] = 0; }\n"; }

void test_errors()
{
    auto&& h = get_handle();
    EXPECT(throws([&] {
        h.AddKernel("GEMM", "", WriteError(), "write", {1, 1, 1}, {1, 1, 1}, "");
    }));
    try
    {
        h.AddKernel("GEMM", "", WriteError(), "write", {1, 1, 1}, {1, 1, 1}, "");
    }
    catch(miopen::Exception& e)
    {
        EXPECT(!std::string(e.what()).empty());
    }
}

std::string WriteNop() { return "__kernel void write(__global int* data) {}\n"; }

void test_warnings()
{
    auto&& h = get_handle();
#if MIOPEN_BUILD_DEV
    EXPECT(throws([&] { h.AddKernel("GEMM", "", WriteNop(), "write", {1, 1, 1}, {1, 1, 1}, ""); }));
#else
    (void)h; // To silence warnings.
#endif
}

void test_arch_name()
{
    auto&& h        = get_handle();
    auto known_arch = {"gfx906", "gfx900", "gfx803"};
    auto this_arch  = h.GetDeviceName();
    EXPECT(std::any_of(
        known_arch.begin(), known_arch.end(), [&](std::string arch) { return arch == this_arch; }));
}

int main()
{
    test_multithreads();
    test_errors();
    test_arch_name();
// Warnings currently dont work in opencl
#if !MIOPEN_BACKEND_OPENCL
    test_warnings();
#endif
}
