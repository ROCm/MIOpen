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

#ifndef GUARD_GET_HANDLE_HPP
#define GUARD_GET_HANDLE_HPP

#include <miopen/handle.hpp>
#include <thread>

#ifndef MIOPEN_TEST_USE_GLOBAL_HANDLE
#define MIOPEN_TEST_USE_GLOBAL_HANDLE 1
#endif

#if MIOPEN_TEST_USE_GLOBAL_HANDLE

inline miopen::Handle& get_handle()
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static miopen::Handle h{};
    static const std::thread::id id = std::this_thread::get_id();
    if(std::this_thread::get_id() != id)
    {
        std::cout << "Cannot use handle across multiple threads\n";
        std::abort();
    }
    return h;
}

#else

inline miopen::Handle get_handle() { return miopen::Handle{}; }

#endif

inline miopen::Handle get_handle_with_stream(const miopen::Handle& h)
{
    return miopen::Handle{h.GetStream()};
}

#endif
