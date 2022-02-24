/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_TIMER_HPP
#define GUARD_MIOPEN_TIMER_HPP

#include <chrono>
#include <cassert>
#include <miopen/handle.hpp>

#define WALL_CLOCK inflags.GetValueInt("wall")

#define START_TIME \
    if(WALL_CLOCK) \
    {              \
        t.start(); \
    }

#define STOP_TIME  \
    if(WALL_CLOCK) \
    {              \
        t.stop();  \
    }

class Timer
{
public:
    Timer(){};
    void start(const bool enabled = true)
    {
        if(!enabled)
            return;
        st = std::chrono::steady_clock::now();
    }
    void stop(const bool enabled = true)
    {
        if(!enabled)
            return;
        et = std::chrono::steady_clock::now();
    }
    float gettime_ms()
    {
        return std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(et - st)
            .count();
    }

private:
    std::chrono::time_point<std::chrono::steady_clock> st;
    std::chrono::time_point<std::chrono::steady_clock> et;
};

class Timer2
{
public:
    Timer2(){};
    void start(const bool enabled = true)
    {
        if(!enabled)
            return;
        st     = std::chrono::steady_clock::now();
        paused = 0.0f;
        state  = Started;
    }
    void pause(const bool enabled = true)
    {
        if(!enabled)
            return;
        assert(state == Started || state == Resumed);
        pst   = std::chrono::steady_clock::now();
        state = Paused;
    }
    void resume(const bool enabled = true)
    {
        if(!enabled)
            return;
        assert(state == Paused);
        const auto pet = std::chrono::steady_clock::now();
        paused +=
            std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(pet - pst).count();
        state = Resumed;
    }
    void stop(const bool enabled = true)
    {
        if(!enabled)
            return;
        assert(state != Stopped);
        if(state == Paused)
            resume();
        et    = std::chrono::steady_clock::now();
        state = Stopped;
    }
    float gettime_ms()
    {
        assert(state == Stopped);
        return std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(et - st)
                   .count() -
               paused;
    }

private:
    enum
    {
        Started,
        Paused,
        Resumed,
        Stopped
    } state      = Stopped;
    float paused = 0.0f;
    std::chrono::time_point<std::chrono::steady_clock> pst;
    std::chrono::time_point<std::chrono::steady_clock> st;
    std::chrono::time_point<std::chrono::steady_clock> et;
};

#endif // GUARD_MIOPEN_TIMER_HPP
