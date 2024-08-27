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
        capture();
    }
    float gettime_ms()
    {
        return std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(et - st)
            .count();
    }
    float interim_time_ms()
    {
        capture();
        return gettime_ms();
    }

private:
    void capture() { et = std::chrono::steady_clock::now(); }
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

class RNNCombTimeLoger
{
public:
    RNNCombTimeLoger(hipStream_t main_stream, size_t size, int mode)
        : stream(main_stream), clockMode(static_cast<ClockMode>(mode))
    {
        if(clockMode != ClockMode::Disabled)
        {
            hostTimePerLaunch.reserve(size);

            startEvent.reserve(size);
            endEvent.reserve(size);
            for(auto i = size; i > 0; --i)
            {
                startEvent.push_back(miopen::make_hip_event());
                endEvent.push_back(miopen::make_hip_event());
            }
        }
    }

    void Start()
    {
        if(clockMode == ClockMode::Disabled)
            return;

        auto launchCount = hostTimePerLaunch.size();

        if(launchCount >= startEvent.size())
        {
            printf("Executed more iterations than planned\n");
            return;
        }

        hipEventRecord(startEvent[launchCount].get(), stream);
        st = std::chrono::steady_clock::now();
    }
    void StopAndPush()
    {
        if(clockMode == ClockMode::Disabled)
            return;

        auto end         = std::chrono::steady_clock::now();
        auto launchCount = hostTimePerLaunch.size();

        if(launchCount >= endEvent.size())
        {
            printf("Executed more iterations than planned\n");
            return;
        }
        hipEventRecord(endEvent[launchCount].get(), stream);

        if(clockMode == ClockMode::OldWallClock)
        {
            std::chrono::time_point<std::chrono::steady_clock> st2 =
                std::chrono::steady_clock::now();

            hipEventSynchronize(endEvent[launchCount].get());

            std::chrono::time_point<std::chrono::steady_clock> end2 =
                std::chrono::steady_clock::now();

            hostTimePerLaunch.push_back(
                std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(end - st)
                    .count() +
                std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(end2 - st2)
                    .count());
        }
        else
        {
            if(clockMode == ClockMode::SeparateClocksSynced)
                hipEventSynchronize(endEvent[launchCount].get());

            hostTimePerLaunch.push_back(
                std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(end - st)
                    .count());
        }
    }

    void Print() const
    {
        auto n_iter = hostTimePerLaunch.size();
        if(clockMode == ClockMode::Disabled || n_iter == 0)
            return;

        float gpu_avg  = 0.0f;
        float host_avg = 0.0f;
        float gpu_time = 0.0f;

        if(clockMode == ClockMode::SeparateClocksNotSynced)
            hipEventSynchronize(endEvent[n_iter - 1].get());

        for(auto i = 0ull; i < n_iter; ++i)
        {
            hipEventElapsedTime(&gpu_time, startEvent[i].get(), endEvent[i].get());

            if(clockMode != ClockMode::OldWallClock)
            {
                printf("launch#%llu, host_time= %f ms, gpu_time= %f ms\n",
                       i,
                       hostTimePerLaunch[i],
                       gpu_time);
            }

            if(i > 0)
            {
                gpu_avg += gpu_time;
                host_avg += hostTimePerLaunch[i];
            }
        }

        if(n_iter == 1)
            hipEventElapsedTime(&gpu_time, startEvent[0].get(), endEvent[0].get());

        printf("GPU Kernel Time Elapsed: %f ms\n", n_iter > 1 ? gpu_avg / (n_iter - 1) : gpu_time);
        printf("Wall-clock Time Elapsed: %f ms\n",
               n_iter > 1 ? host_avg / (n_iter - 1) : hostTimePerLaunch[0]);
    }

    enum class ClockMode
    {
        Disabled                = 0,
        OldWallClock            = 1,
        SeparateClocksSynced    = 2,
        SeparateClocksNotSynced = 3
    };

private:
    std::vector<float> hostTimePerLaunch;

    std::vector<miopen::HipEventPtr> startEvent;
    std::vector<miopen::HipEventPtr> endEvent;

    hipStream_t stream;
    std::chrono::time_point<std::chrono::steady_clock> st;

    ClockMode clockMode;
};

#endif // GUARD_MIOPEN_TIMER_HPP
