/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#ifndef MIOPEN_GUARD_MLOPEN_SOLVER_ID_HPP
#define MIOPEN_GUARD_MLOPEN_SOLVER_ID_HPP

#include <miopen/logger.hpp>
#include <miopen/conv_algo_name.hpp>

#include <cstdint>
#include <unordered_map>

namespace miopen {
namespace solver {

struct AnySolver;

struct Id
{
    static constexpr uint64_t invalid_value = 0;

    Id() = default;
    Id(uint64_t value_);
    Id(const std::string& str);
    Id(const char* str);

    std::string ToString() const;
    AnySolver GetSolver() const;
    std::string GetAlgo(conv::Direction dir) const;

    bool IsValid() const { return is_valid; }
    uint64_t Value() const { return value; }
    bool operator==(const Id& other) const
    {
        if(!is_valid && !other.is_valid)
            return true; // invalids are equal regardless of their values
        return value == other.value && is_valid == other.is_valid;
    }
    bool operator!=(const Id& other) const { return !(*this == other); }

    static solver::Id gemm()
    {
        static const auto value = solver::Id{"gemm"};
        return value;
    }

    static solver::Id fft()
    {
        static const auto value = solver::Id{"fft"};
        return value;
    }

    private:
    uint64_t value = invalid_value;
    bool is_valid  = false;
};

} // namespace solver
} // namespace miopen

#endif
