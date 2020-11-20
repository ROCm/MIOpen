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
#ifndef GUARD_MIOPEN_TIMER_HPP_
#define GUARD_MIOPEN_TIMER_HPP_

#include <miopen/logger.hpp>
#include <chrono>

namespace miopen {

class Timer
{
    public:
    Timer(){};
    void start() { st = std::chrono::steady_clock::now(); }
    float elapsed_ms()
    {
        capture();
        return std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(et - st)
            .count();
    }

    private:
    void capture() { et = std::chrono::steady_clock::now(); }
    std::chrono::time_point<std::chrono::steady_clock> st;
    std::chrono::time_point<std::chrono::steady_clock> et;
};

class CompileTimer
{
#if MIOPEN_BUILD_DEV
    Timer timer;
#endif
    public:
    CompileTimer()
    {
#if MIOPEN_BUILD_DEV
        timer.start();
#endif
    }
    void Log(const std::string& s1, const std::string& s2 = {})
    {
#if MIOPEN_BUILD_DEV
        MIOPEN_LOG_I2(
            s1 << (s2.empty() ? "" : " ") << s2 << " Compile Time, ms: " << timer.elapsed_ms());
#else
        (void)s1;
        (void)s2;
#endif
    }
};

class FunctionTimer{
 public:
  FunctionTimer(std::string name)
      : m_name(std::move(name)),
        m_beg(std::chrono::high_resolution_clock::now()) { }
  ~FunctionTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - m_beg);
    std::cout << m_name << " : " << dur.count() << " musec\n";
  }
 private:
  std::string m_name;
  std::chrono::time_point<std::chrono::high_resolution_clock> m_beg;
};

#if MIOPEN_TIME_FUNCTIONS
#define MIOPEN_FUNC_TIMER                               \
        miopen::FunctionTimer miopen_timer(miopen::LoggingParseFunction(__func__, __PRETTY_FUNCTION__))
#else
#define MIOPEN_FUNC_TIMER
#endif


} // namespace miopen

#endif // GUARD_MIOPEN_TIMER_HPP_
