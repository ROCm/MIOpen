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
#pragma once

#include <random>
#include <cassert>
#include <vector>
#include <chrono>

struct LazyExponentialBackoff
{
    using value_type = int;
    using reference  = value_type const&;
    using pointer    = value_type const*;
    std::random_device dev;
    std::mt19937 gen;
    std::uniform_int_distribution<> dis;
    int seq_idx;
    std::vector<int> buf;
    int max_rand;
    int base;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time;
    std::chrono::seconds secs;
    const int max_buf_sz = 20;
    int buf_idx          = 0;

public:
    LazyExponentialBackoff(int _max_rand              = 10,
                           int _base                  = 2,
                           std::chrono::seconds _secs = std::chrono::seconds(30))
        : gen(std::mt19937{dev()}), seq_idx(0), max_rand(_max_rand), base(_base), secs(_secs)

    {
        dis = std::uniform_int_distribution<>(0, max_rand);
        buf.resize(max_buf_sz);
        for(auto idx = 0; idx < max_buf_sz; idx++)
            buf[idx] = dis(gen);
        end_time = std::chrono::high_resolution_clock::now() + secs;
    }
    explicit operator bool() const
    {
        return !(std::chrono::high_resolution_clock::now() > end_time);
    }

    value_type operator*()
    {
        if(buf_idx >= buf.size())
        {
            for(auto idx = 0; idx < max_buf_sz; idx++)
                buf[idx] = dis(gen);
            buf_idx = 0;
        }
        if(seq_idx > 0 && seq_idx < max_rand)
            return std::pow(base, buf[buf_idx++] % seq_idx);
        else
            return std::pow(base, buf[buf_idx++]);
    } // number from the random seq
};
