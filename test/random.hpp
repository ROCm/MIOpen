/*******************************************************************************
*
* MIT License
*
* Copyright (c) 2021 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_TEST_RANDOM_HPP
#define GUARD_MIOPEN_TEST_RANDOM_HPP

#include <cstdlib>

// template <typename T>
// static T FRAND(void)
//{
//    double d = static_cast<double>(rand() / (static_cast<double>(RAND_MAX)));
//    return static_cast<T>(d);
//}

/// Basically, this is a copy of driver/random.hpp. Why:
/// We want to have the same functionality as implemented in driver/random.hpp,
/// But we want this functionality to be independent, so changes in tests won't affect the driver
/// and vice versa. This independency could be important, because, for example, the driver
/// implements its own cache of verification data and change or GET_RAND() would break it.

static int GET_RAND()
{
    return rand(); // NOLINT (concurrency-mt-unsafe)
}

// template <typename T>
// static T RAN_GEN(T A, T B)
//{
//    T r = (FRAND<T>() * (B - A)) + A;
//    return r;
//}

#endif
