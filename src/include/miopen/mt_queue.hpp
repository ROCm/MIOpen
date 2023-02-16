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

#pragma once

#include <queue>
#include <condition_variable>
#include <mutex>

template <typename T>
class ThreadSafeQueue
{
    std::mutex mutex;
    std::condition_variable cond_var;
    std::queue<T> queue;

public:
    void push(T&& item)
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            queue.push(item);
        }

        cond_var.notify_one();
    }

    T& front()
    {
        std::unique_lock<std::mutex> lock(mutex);
        cond_var.wait(lock, [&] { return !queue.empty(); });
        return queue.front();
    }

    void pop()
    {
        std::lock_guard<std::mutex> lock(mutex);
        queue.pop();
    }
};
