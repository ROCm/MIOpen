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
#include "miopen/tensor_ops.hpp"
#include "tensor_holder.hpp"
#include "gtest/conv_tensor_gen.hpp"
#include "get_handle.hpp"

#include <chrono>

#define TEST_SAMPLES 10

const std::vector<size_t> GetInput() { return {4, 4, 1115, 1115}; }

// set the gpu_ptr(device)
template <typename T>
long long testSetTensor(void* gpu_ptr, const tensor<T>& input)
{
    float zero               = 0.f;
    auto&& handle            = get_handle();
    long long total_durtaion = 0;

    for(int i = 0; i < TEST_SAMPLES; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        SetTensor(handle, input.desc, gpu_ptr, &zero);
        // End time
        auto end = std::chrono::high_resolution_clock::now();

        // Calculating total duration
        total_durtaion +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    return total_durtaion / TEST_SAMPLES;
}

// copy input data (host) to gpu_ptr(device)
template <typename T>
long long testHIPMemcpy(void* gpu_ptr, const tensor<T>& input)
{
    long long total_durtaion = 0;
    for(int i = 0; i < TEST_SAMPLES; ++i)
    {
        auto start        = std::chrono::high_resolution_clock::now();
        auto ret_cpy_hTod = hipMemcpy(static_cast<void*>(gpu_ptr),
                                      input.data.data(),
                                      sizeof(T) * input.data.size(),
                                      hipMemcpyHostToDevice);
        auto end          = std::chrono::high_resolution_clock::now();

        if(ret_cpy_hTod != hipSuccess)
        {
            std::cout << "\n\nhipMemcpy from host to device : Error code :" << ret_cpy_hTod
                      << "\n\n";
            exit(1);
        }
        total_durtaion +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    return total_durtaion / TEST_SAMPLES;
}

// set the gpu_ptr(device)
template <typename T>
long long testHIPMemset(void* gpu_ptr, const tensor<T>& input)
{
    long long total_durtaion = 0;
    for(int i = 0; i < TEST_SAMPLES; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto ret_cpy_hTod =
            hipMemset(static_cast<void*>(gpu_ptr), 0, sizeof(T) * input.data.size());
        auto end = std::chrono::high_resolution_clock::now();

        if(ret_cpy_hTod != hipSuccess)
        {
            std::cout << "\n\nhipMemcpy from host to device : Error code :" << ret_cpy_hTod
                      << "\n\n";
            exit(1);
        }
        total_durtaion +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    return total_durtaion / TEST_SAMPLES;
}

int main()
{
    std::cout << "\nStart benchmarking ..\n";
    std::map<std::string, long long> perf_data;

    using T                            = float;
    miopenTensorLayout_t tensor_layout = miopenTensorNCHW;
    tensor<T> input;
    tensor<T> output;
    void* gpu_ptr;

    input = tensor<T>{tensor_layout, GetInput()};
    input.generate(GenData<T>{});
    output.data.resize(input.data.size());

    // allocate memory in GPU
    auto ret_malloc = hipMalloc(static_cast<void**>(&gpu_ptr), sizeof(T) * input.data.size());
    if(ret_malloc != hipSuccess)
    {
        std::cout << "\n\nCannot allocate memory : Error code :" << ret_malloc << "\n\n";
        exit(1);
    }
    // do the performance
    perf_data["hipMemcpy"]     = testHIPMemcpy(gpu_ptr, input);
    perf_data["hipMemset"]     = testHIPMemset(gpu_ptr, input);
    perf_data["testSetTensor"] = testSetTensor(gpu_ptr, input);

    for(auto it : perf_data)
    {
        std::cout << it.first << " = " << it.second << "\n";
    }
    hipFree(gpu_ptr);
    std::cout << "\nDone benchmarking\n";
}
