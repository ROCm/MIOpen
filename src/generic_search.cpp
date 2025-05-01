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
#include <chrono>

namespace miopen {
namespace solver {
namespace debug {

// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
static std::optional<std::size_t> tuning_iterations_limit;

TuningIterationScopedLimiter::TuningIterationScopedLimiter(std::size_t new_limit)
    : old_limit(tuning_iterations_limit)
{
    tuning_iterations_limit = new_limit;
}

TuningIterationScopedLimiter::~TuningIterationScopedLimiter()
{
    tuning_iterations_limit = old_limit;
}
} // namespace debug

std::size_t GetTuningIterationsMax()
{
    if(debug::tuning_iterations_limit)
        return *debug::tuning_iterations_limit;
    return env::value(MIOPEN_DEBUG_TUNING_ITERATIONS_MAX);
}

std::chrono::milliseconds GetTuningTimeMax()
{
    return std::chrono::milliseconds{env::value(MIOPEN_TUNING_TIME_MS_MAX)};
}

std::size_t GetTuningThreadsMax() { return env::value(MIOPEN_COMPILE_PARALLEL_LEVEL); }

std::size_t GetTuningPatience() { return env::value(MIOPEN_TUNING_PATIENCE); }

} // namespace solver
} // namespace miopen
