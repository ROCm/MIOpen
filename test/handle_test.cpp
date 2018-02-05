
#include <miopen/handle.hpp>
#include "get_handle.hpp"
#include <vector>
#include <thread>
#include "test.hpp"

std::string Write2s()
{
    return "__kernel void write(__global int* data) { data[get_global_id(0)] *= 2; }\n";
}

void run2s(std::size_t n)
{
    auto&& h = get_handle();
    std::vector<int> data_in(n, 1);
    auto data_dev = h.Write(data_in);

    h.AddKernel("GEMM", "", Write2s(), "write", {n, 1, 1}, {n, 1, 1}, "")(data_dev.get());
    std::fill(data_in.begin(), data_in.end(), 2);

    auto data_out = h.Read<int>(data_dev, n);
    CHECK(data_out == data_in);
}

void test_multithreads()
{
    std::thread([&] { run2s(16); }).join();
    std::thread([&] { run2s(32); }).join();
    std::thread([&] { std::thread([&] { run2s(64); }).join(); }).join();
    run2s(4);
}

std::string WriteNop()
{
    return "__kernel void write(__global int* data) {}\n";
}

void test_warnings()
{
    auto&& h = get_handle();
#if MIOPEN_BUILD_DEV
    EXPECT(throws([&] { h.AddKernel("GEMM", "", WriteNop(), "write", {1, 1, 1}, {1, 1, 1}, ""); }));
#endif
}

int main()
{
    test_multithreads();
    test_warnings();
}
