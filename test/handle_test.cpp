
#include <miopen/handle.hpp>
#include "get_handle.hpp"
#include <vector>
#include <thread>
#include "test.hpp"

std::string Write2s()
{
    return "__kernel void write(__global int* data) { data[get_global_id(0)] *= 2; }\n";
}

void run(std::size_t n)
{
    auto&& h = get_handle();
    std::vector<int> data_in(n, 1);
    auto data_dev = h.Write(data_in);

    h.GetKernel("GEMM", "", Write2s(), "write", {n, 1, 1}, {n, 1, 1}, "")(data_dev.get());
    std::fill(data_in.begin(), data_in.end(), 2);

    auto data_out = h.Read<int>(data_dev, n);
    CHECK(data_out == data_in);
}

int main()
{
    std::thread([&] { run(16); }).join();
    std::thread([&] { run(32); }).join();
    std::thread([&] { std::thread([&] { run(64); }).join(); }).join();
    run(4);
}
