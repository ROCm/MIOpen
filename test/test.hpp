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
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>

#ifndef GUARD_TEST_TEST_HPP_
#define GUARD_TEST_TEST_HPP_

inline void failed(const char* msg, const char* file, int line)
{
    auto ss = std::ostringstream{};
    ss << "FAILED: " << msg << ": " << file << ": " << line << std::endl;
    std::cout << ss.str();
}

[[gnu::noreturn]] inline void failed_abort(const char* msg, const char* file, int line)
{
    failed(msg, file, line);
    std::abort();
}

template <class TLeft, class TOp, class TRight>
inline void expect_op(const TLeft& left,
                      const TOp& op,
                      const TRight& right,
                      const char* left_s,
                      const char* op_s,
                      const char* riglt_s,
                      const char* file,
                      int line)
{
    if(op(left, right))
        return;

    auto ss = std::ostringstream{};
    ss << "FAILED: " << left_s << "(" << left << ") " << op_s << " " << riglt_s << "(" << right
       << "): " << file << ':' << line << std::endl;
    std::cout << ss.str();
    std::abort();
}

#define CHECK(...)                                    \
    do                                                \
    {                                                 \
        if(!(__VA_ARGS__))                            \
            failed(#__VA_ARGS__, __FILE__, __LINE__); \
                                                      \
    } while(false)

#define EXPECT(...)                                         \
    do                                                      \
    {                                                       \
        if(!(__VA_ARGS__))                                  \
            failed_abort(#__VA_ARGS__, __FILE__, __LINE__); \
                                                            \
    } while(false)

#define EXPECT_OP(LEFT, OP, RIGHT)                                 \
    expect_op((LEFT),                                              \
              [](const auto& l, const auto& r) { return l OP r; }, \
              (RIGHT),                                             \
              #LEFT,                                               \
              #OP,                                                 \
              #RIGHT,                                              \
              __FILE__,                                            \
              __LINE__)
#define EXPECT_EQUAL(LEFT, RIGHT) EXPECT_OP(LEFT, ==, RIGHT)
#define STATUS(...) EXPECT((__VA_ARGS__) == 0)

#define FAIL(...) failed(__VA_ARGS__, __FILE__, __LINE__)

template <class F>
bool throws(F f)
{
    try
    {
        f();
        return false;
    }
    catch(...)
    {
        return true;
    }
}

template <class F, class Exception>
bool throws(F f, std::string msg = "")
{
    try
    {
        f();
        return false;
    }
    catch(const Exception& ex)
    {
        return std::string(ex.what()).find(msg) != std::string::npos;
    }
}

template <class T>
void run_test()
{
    T t = {};
    t.run();
}

#endif
