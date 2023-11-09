/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#include <miopen/generic_search.hpp>
#include <miopen/generic_search_controls.hpp>

#include <cstddef>
#include <limits>
#include <chrono>

namespace miopen {
namespace solver {

std::size_t GetTuningIterationsMax()
{
    return !IsUnset(MIOPEN_DEBUG_TUNING_ITERATIONS_MAX{})
               ? Value(MIOPEN_DEBUG_TUNING_ITERATIONS_MAX{})
               : std::numeric_limits<std::size_t>::max();
}

std::chrono::milliseconds GetTuningTimeMax()
{
    static const auto fallback =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::hours{2});
    static const auto res = !IsUnset(MIOPEN_TUNING_TIME_MS_MAX{})
                                ? std::chrono::milliseconds{Value(MIOPEN_TUNING_TIME_MS_MAX{})}
                                : std::chrono::milliseconds{fallback.count()};
    return res;
}

std::size_t GetTuningThreadsMax()
{
#if MIOPEN_USE_COMGR
    const auto def_max = 1; // COMGR is not parallelizable
#else
    const int def_max = std::thread::hardware_concurrency() / 2;
#endif
    return !IsUnset(MIOPEN_COMPILE_PARALLEL_LEVEL{}) ? Value(MIOPEN_COMPILE_PARALLEL_LEVEL{})
                                                     : def_max;
}

} // namespace solver
} // namespace miopen
