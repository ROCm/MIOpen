/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include <gtest/gtest.h>
#include <miopen/mt_queue.hpp>
#include <thread>
#include <chrono>

#include "random.hpp"

static std::atomic<int> num_prod{};

static const auto total_producers = std::thread::hardware_concurrency();
const auto data_len               = 100;

template <typename T>
using data_t = std::vector<T>;

template <typename T>
void producer(int thread_idx, data_t<T>& common_data, ThreadSafeQueue<T>& comp_queue)
{
    prng::reset_seed(thread_idx);
    for(auto idx = thread_idx; idx < data_len; idx += total_producers)
    {
        comp_queue.push(std::move(common_data.at(idx)));
        num_prod.fetch_add(1, std::memory_order_relaxed);
        std::this_thread::sleep_for(std::chrono::milliseconds(prng::gen_0_to_B(100) + 1));
    }
}

TEST(CPU_UtilMultiThreadQueue_NONE, Basic)
{
    ThreadSafeQueue<int> comp_queue;
    int num_cons = 0;
    data_t<int> common_data;
    for(auto idx = 0; idx < data_len; ++idx)
        common_data.emplace_back(idx);

    std::vector<std::thread> producers;
    for(int idx = 0; idx < total_producers; idx++)
    {
        producers.emplace_back(producer<int>, idx, std::ref(common_data), std::ref(comp_queue));
    }

    for(auto idx = 0; idx < data_len; ++idx)
    {
        auto res = comp_queue.pop();
        std::cerr << res << std::endl;
        num_cons++;
    }

    for(auto& prod : producers)
        prod.join();

    std::cout << "Stage 2" << std::endl;
    for(const auto& tmp : common_data)
        std::cout << tmp << std::endl;
    EXPECT_EQ(num_prod, num_cons);
}
