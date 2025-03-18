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

#include <miopen/config.hpp>
#include <miopen/logger.hpp>
#include <miopen/conv_algo_name.hpp>

#include <cstdint>

namespace miopen {

struct ForceInit
{
};

namespace solver {

struct AnySolver;
struct SolverBase;

enum class Primitive
{
    Invalid,
    Convolution,
    Activation,
    Batchnorm,
    Bias,
    Fusion,
    Pooling,
    Normalization,
    Reduce,
    Cat,
    Mha,
    Softmax,
    Adam,
    Item,
    RoPE,
    ReLU,
    Kthvalue,
    SoftMarginLoss,
    MultiMarginLoss
};

struct MIOPEN_INTERNALS_EXPORT Id
{
    static constexpr uint64_t invalid_value = 0;

    Id() = default;
    Id(uint64_t value_);
    Id(ForceInit, uint64_t value_);
    Id(const std::string& str);
    Id(const char* str);

    std::string ToString() const;
    AnySolver GetSolver() const;
    const SolverBase* GetSolverBase() const;
    std::string GetAlgo(conv::Direction dir) const;
    miopenConvAlgorithm_t GetAlgo() const;
    Primitive GetPrimitive() const;

    bool IsValid() const { return is_valid; }
    uint64_t Value() const { return value; }
    bool operator==(const Id& other) const
    {
        if(!is_valid && !other.is_valid)
            return true; // invalids are equal regardless of their values
        return value == other.value && is_valid == other.is_valid;
    }
    bool operator!=(const Id& other) const { return !(*this == other); }

private:
    uint64_t value = invalid_value;
    bool is_valid  = false;
};

MIOPEN_INTERNALS_EXPORT const std::vector<Id>& GetSolversByPrimitive(Primitive primitive);

} // namespace solver
} // namespace miopen

#endif
